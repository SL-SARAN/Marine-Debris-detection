# Marine Debris Detection System Architecture

## 1. System Purpose

The system processes side-scan sonar data to detect marine debris, ghost nets, shipwrecks, subsea infrastructure, and other acoustic anomalies. It combines acoustic image processing, YOLO-based object detection, acoustic shadow verification, sonar geometry, navigation data, and GIS reporting.

The architecture is designed to address:

- Acoustic speckle noise and low contrast
- Variable sonar resolution and range distortion
- Acoustic highlight and shadow interpretation
- Vehicle motion from heave, pitch, and roll
- Missing or degraded navigation and scan lines
- Accurate target measurement and georeferencing

## 2. High-Level System Architecture

```mermaid
flowchart LR
    A[Raw XTF / Sonar Image] --> B[Data Ingestion]
    B --> C[Sonar Packet and Image Extraction]
    C --> D[Telemetry and Input QC]
    D --> E[Motion and Geometric Correction]
    E --> F[TVG / Radiometric Correction]
    F --> G[Acoustic Conditioning]
    G --> H[AI Object Detection]
    H --> I[Acoustic Physics Verification]
    I --> J[Measurement and Height Estimation]
    J --> K[Georeferencing]
    K --> L[Quality-Aware Reporting]
    L --> M[GIS Dashboard and Survey Reports]
```

### Main stages

1. **Data ingestion** accepts raw XTF files or sonar image inputs.
2. **Sonar extraction** decodes channels, pings, timestamps, and sensor telemetry.
3. **Telemetry and input QC** validates packets, timestamps, channels, GPS, attitude, and sensor values.
4. **Motion and geometric correction** uses navigation and attitude data to create a corrected sonar swath.
5. **TVG / radiometric correction** compensates for range-dependent acoustic intensity loss.
6. **Acoustic conditioning** reduces speckle and improves target-to-background contrast.
7. **AI object detection** identifies candidate targets at multiple scales.
8. **Acoustic physics verification** confirms highlight-shadow relationships.
9. **Measurement and height estimation** calculates ground range, dimensions, and estimated height.
10. **Georeferencing** converts sonar detections into WGS84 coordinates.
11. **Quality-aware reporting** publishes detections, confidence, uncertainty, and processing quality.

## 3. Detailed Processing Architecture

```mermaid
flowchart TD
    A[Raw XTF / Sonar Image] --> B{Input Type}

    B -->|XTF| C[pyxtf XTF Decoder]
    B -->|Image| D[Image Input Adapter]
    C --> E[Ping Records + Sonar Channels + Telemetry]
    D --> E

    E --> F[Stage 1: Input QC]
    F --> F1[Packet Integrity]
    F --> F2[Timestamp Validation]
    F --> F3[Channel and Sample Validation]
    F --> F4[GPS Validation]
    F --> F5[IMU and Sensor Validation]
    F --> F6[Missing Ping Detection]

    F1 --> G[Navigation / Motion Data]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G
    F6 --> G

    G --> H[Motion Compensation]
    H --> H1[Heave Compensation]
    H --> H2[Pitch Correction]
    H --> H3[Roll Correction]
    H --> H4[Heading Alignment]
    H --> H5[Scan-Line Gap Handling]

    H1 --> I[Geometric Correction]
    H2 --> I
    H3 --> I
    H4 --> I
    H5 --> I

    I --> J[Corrected Sonar Swath]
    J --> K[Stage 2: Processing QC]
    K --> K1[Excessive Motion Check]
    K --> K2[Interpolation Percentage]
    K --> K3[Geometric Distortion Check]
    K --> K4[Corrected Image Quality]

    K --> L[TVG / Radiometric Correction]
    L --> M[Acoustic Conditioning]
    M --> M1[Noise Estimation]
    M1 --> M2[Wavelet Preprocessing]
    M2 --> M3[Enhanced Lee Filtering]
    M3 --> M4[CLAHE Contrast Enhancement]
    M4 --> M5[Background Normalization]
    M5 --> M6[Slant-to-Ground Rectification]

    M6 --> N[Prepared Sonar Image]
    N --> O[Multi-Scale Inference]
    O --> O1[Full-Image YOLO Inference]
    O --> O2[Tiled YOLO Inference]
    O1 --> P[Cross-Scale Box Merging / NMS]
    O2 --> P
    P --> Q[Candidate Targets]

    Q --> R[Acoustic Physics Verification]
    R --> R1[Highlight Segmentation]
    R1 --> R2[Shadow Segmentation]
    R2 --> R3[Geometry-Aware Shadow Direction]
    R3 --> R4[Highlight-Shadow Association]
    R4 --> R5[Shadow Length Measurement]

    R5 --> S[Stage 3: Detection QC]
    S --> S1[Detector Confidence]
    S --> S2[Shadow Consistency]
    S --> S3[Physical Plausibility]
    S --> S4[Image and Telemetry Quality]

    S --> T[Composite Confidence Score]
    T --> U[Acoustic Mensuration]
    U --> U1[Ground Range]
    U --> U2[Object Length and Width]
    U --> U3[Shadow Length]
    U --> U4[Object Height Estimation]
    U --> U5[Measurement Uncertainty]

    U --> V[Georeferencing Inputs]
    V --> V1[GPS / Dead Reckoning]
    V --> V2[Navigation Timestamp]
    V --> V3[Heading]
    V --> V4[Pitch and Roll]
    V --> V5[Heave / Altitude]
    V --> V6[Sonar Mounting Geometry]
    V --> V7[Slant and Ground Range]

    V1 --> W[Target Position Calculation]
    V2 --> W
    V3 --> W
    V4 --> W
    V5 --> W
    V6 --> W
    V7 --> W
    W --> X[WGS84 Coordinates + Position Uncertainty]

    F --> Y[Quality State]
    K --> Y
    S --> Y
    X --> Y
    Y --> Y1[GOOD]
    Y --> Y2[DEGRADED]
    Y --> Y3[INVALID]

    T --> Z[Quality-Aware Report Record]
    U --> Z
    X --> Z
    Y --> Z
    Z --> AA[Annotated Sonar Image]
    Z --> AB[GIS Dashboard]
    Z --> AC[CSV / JSON / GeoJSON]
    Z --> AD[Survey PDF]
```

## 4. Navigation and Motion Data Flow

Navigation data must not terminate at validation. It must directly control swath reconstruction and later georeferencing.

```mermaid
flowchart TD
    A[Navigation / Motion Data] --> B[GPS Validation]
    A --> C[IMU Validation]
    A --> D[Telemetry QC]

    B --> E[Validated Navigation State]
    C --> E
    D --> E

    E --> F[Motion Compensation]
    F --> F1[Heave]
    F --> F2[Pitch]
    F --> F3[Roll]
    F --> F4[Heading]

    F1 --> G[Geometric Correction]
    F2 --> G
    F3 --> G
    F4 --> G

    G --> H[Corrected Sonar Swath]
    H --> I[TVG / Radiometric Correction]
    I --> J[Acoustic Conditioning]
    J --> K[AI Detection]
    K --> L[Physics Verification]
    L --> M[Mensuration and Georeferencing]
```

## 5. Acoustic Conditioning Flow

The initial conditioning path uses Enhanced Lee filtering. Wiener and Frost filtering can be evaluated later as alternative methods, but they are not chained automatically with Lee filtering.

```mermaid
flowchart LR
    A[Corrected Sonar] --> B[TVG / Radiometric Correction]
    B --> C[Noise Estimation]
    C --> D[Wavelet Preprocessing]
    D --> E[Enhanced Lee Filtering]
    E --> F[CLAHE]
    F --> G[Background Normalization]
    G --> H[Slant-to-Ground Rectification]
    H --> I[Prepared Sonar Image]
```

## 6. Acoustic Physics Verification

The verifier checks more than detector confidence. It determines whether the candidate has the expected acoustic structure of a physical object.

```mermaid
flowchart TD
    A[Candidate Target] --> B{Bright Highlight?}
    B -->|No| C[Reject or Degrade Confidence]
    B -->|Yes| D{Corresponding Dark Shadow?}
    D -->|No| C
    D -->|Yes| E[Geometry-Aware Shadow Direction Check]
    E -->|Invalid| C
    E -->|Valid| F[Highlight-Shadow Association]
    F --> G[Shadow Length Measurement]
    G --> H[Physical Plausibility Check]
    H --> I[Physics Confidence]
```

Shadow direction is determined from sonar geometry, port/starboard orientation, nadir position, vehicle heading, and sensor mounting configuration. It is not assumed to always point in a fixed image direction.

## 7. Height Estimation Inputs

Shadow length alone is insufficient for reliable target height estimation. The calculation combines the measured shadow with sonar geometry and altitude information.

```mermaid
flowchart LR
    A[Measured Shadow Length] --> E[Object Height Estimation]
    B[Sonar Altitude] --> E
    C[Ground / Slant Range] --> E
    D[Seafloor Geometry] --> E
    E --> F[Estimated Height + Uncertainty]
```

The geometric model is:

$$
H_t = \frac{L_g H_a}{R_g + L_g}
$$

where $H_t$ is target height, $L_g$ is ground-range shadow length, $H_a$ is sonar altitude, and $R_g$ is ground range to the target.

## 8. Composite Confidence and Quality State

The final result combines three evidence groups:

```mermaid
flowchart TD
    A[Detector Confidence] --> D[Composite Confidence Score]
    B[Physics Confidence] --> D
    C[Data Quality Confidence] --> D
    D --> E[Final Detection Result]

    F[Input QC] --> G[Quality State]
    H[Processing QC] --> G
    I[Detection QC] --> G
    G --> G1[GOOD]
    G --> G2[DEGRADED]
    G --> G3[INVALID]
    G --> E
```

Example result:

```json
{
  "object": "ghost_net",
  "confidence": 0.91,
  "latitude": 11.0123,
  "longitude": 76.9558,
  "quality": "GOOD"
}
```

A degraded result should preserve the detection while explaining its limitations:

```json
{
  "object": "ghost_net",
  "confidence": 0.67,
  "quality": "DEGRADED",
  "quality_reasons": [
    "high_roll",
    "missing_ping_interpolation",
    "estimated_altitude"
  ]
}
```

## 9. Georeferencing Inputs

The sonar pixel-to-coordinate transformation uses the following inputs:

- GPS or dead-reckoned platform position
- Navigation timestamp matched to the detection ping
- Vehicle and towfish heading
- Pitch and roll
- Heave or sonar altitude
- Sonar mounting offsets and orientation
- Port/starboard channel geometry
- Slant-to-ground corrected range

The output includes WGS84 coordinates and an uncertainty estimate rather than coordinates alone.

## 10. Recommended PPT Presentation

### Slide 1: High-Level Architecture

Use the system architecture diagram from Section 2. Explain the pipeline in one sentence:

> Raw sonar data is decoded, motion-corrected, acoustically enhanced, analyzed by AI, verified using acoustic physics, georeferenced, and delivered through GIS reports.

### Slide 2: Detailed Technical Flow

Use the detailed processing diagram from Section 3. Emphasize that navigation and quality information flow into corrected sonar, detection, mensuration, georeferencing, and final reporting.

### Slide 3: Key Innovation

Show the acoustic physics verification diagram from Section 6. Emphasize that a candidate is not accepted using AI confidence alone; its highlight, shadow, direction, and physical plausibility are also evaluated.

### Slide 4: Output

Show the quality-aware result containing:

- Detected object class
- Composite confidence
- Object dimensions and estimated height
- WGS84 coordinates
- Quality state and reasons
- Annotated sonar image and GIS report
