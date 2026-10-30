import os
import time
import cv2
import numpy as np
from pathlib import Path

# Standard class mapping for SSS marine anomaly detection (6 classes)
CLASS_NAMES = {
    0: "ghost_net",
    1: "shipwreck",
    2: "aircraft",
    3: "seafloor_debris",
    4: "pipeline_cable",
    5: "engineering_platform"
}

CLASS_COLORS = {
    0: (255, 229, 0),    # Cyan/Turquoise (BGR) for Ghost Nets
    1: (48, 59, 255),    # Coral Red (BGR) for Shipwrecks
    2: (0, 149, 255),    # Vibrant Amber/Orange (BGR) for Aircraft
    3: (0, 204, 255),    # Golden Yellow (BGR) for General Seafloor Debris
    4: (255, 0, 170),    # Vivid Magenta (BGR) for Subsea Pipelines & Cables
    5: (0, 255, 128)     # Bright Emerald Green (BGR) for Engineering Platforms
}

class SSSObjectDetector:
    """
    Production-grade Side-Scan Sonar Object Detector wrapping YOLOv8n.
    Optimized for low-latency marine drone / AUV edge deployment.
    """
    def __init__(self, model_path=None, conf_thres=0.25, iou_thres=0.45, device=None):
        import torch
        from ultralytics import YOLO

        if model_path is None:
            default_best = Path(__file__).resolve().parent.parent.parent / "models" / "best.pt"
            if default_best.exists():
                model_path = str(default_best)
            else:
                model_path = "yolov8n.pt"

        self.model_path = str(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[SSSObjectDetector] Loading weights from: {self.model_path} on device: {self.device}")
        self.model = YOLO(self.model_path)
        self.class_names = getattr(self.model, "names", CLASS_NAMES)

    def predict(self, image_input, conf=None, iou=None):
        """
        Runs inference on an image array (numpy BGR or RGB) or file path.
        Returns structured detection dictionary.
        """
        conf = conf or self.conf_thres
        iou = iou or self.iou_thres

        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
        else:
            img = image_input.copy()

        h, w = img.shape[:2]
        t0 = time.perf_counter()

        results = self.model.predict(
            source=img,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        r = results[0]
        speed = getattr(r, "speed", {})

        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                cid = clss[i]
                cname = self.class_names.get(cid, f"class_{cid}") if isinstance(self.class_names, dict) else str(self.class_names[cid])
                score = float(scores[i])

                detections.append({
                    "id": i + 1,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(cid),
                    "class_name": cname,
                    "raw_confidence": score,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area_px": (x2 - x1) * (y2 - y1)
                })

        return {
            "detections": detections,
            "total_detected": len(detections),
            "image_size": [w, h],
            "latency": {
                "preprocess_ms": speed.get("preprocess", 0.0),
                "inference_ms": speed.get("inference", total_latency_ms),
                "postprocess_ms": speed.get("postprocess", 0.0),
                "total_pipeline_ms": total_latency_ms
            }
        }

    def draw_detections(self, image, detections, show_confidence=True):
        """
        Renders bounding boxes, highlight anchors, and labels on an image.
        """
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cid = det["class_id"]
            cname = det["class_name"]
            score = det["raw_confidence"]
            color = CLASS_COLORS.get(cid, (0, 255, 0))

            # Bounding rectangle
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label banner
            conf_str = f" {score*100:.1f}%" if show_confidence else ""
            label = f"{cname.upper()}{conf_str}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(vis, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(lh + 6, y1)), color, -1)
            cv2.putText(vis, label, (x1 + 3, max(lh, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return vis
