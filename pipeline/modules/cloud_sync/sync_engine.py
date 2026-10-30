"""
Cloud Sync Engine — non-blocking orchestrator for uploads, connectivity detection,
retry logic with exponential backoff, and periodic background sync.

This engine is STRICTLY ADDITIVE/OPTIONAL. If it fails to initialise (missing Azure
SDK, no credentials, network issues), all local report generation continues
completely unaffected. No detection, preprocessing, confidence scoring, or report
generation code path imports from or depends on this module.

Retry schedule (exponential backoff):
    Attempt 1 →  1 minute
    Attempt 2 →  5 minutes
    Attempt 3 → 15 minutes
    Attempt 4 →  1 hour
    Attempt 5 →  6 hours
    > 5 failures → status = 'failed_manual' (requires manual retry from UI)

Connectivity check: configurable interval (default 3 min, env CLOUD_SYNC_CHECK_INTERVAL_SECONDS).
Manifest pruning: uploaded entries older than 30 days removed automatically (tracking row only).
"""

import os
import time
import zipfile
import threading
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from .manifest import UploadManifest
from .uploader import AzureBlobUploader

# Exponential backoff schedule (seconds) for retry attempts 1-5
RETRY_BACKOFF_SECONDS = [60, 300, 900, 3600, 21600]

# Configurable via environment
DEFAULT_CHECK_INTERVAL = int(os.environ.get("CLOUD_SYNC_CHECK_INTERVAL_SECONDS", "180"))
DEFAULT_PRUNE_DAYS = int(os.environ.get("CLOUD_SYNC_MANIFEST_PRUNE_DAYS", "30"))


class CloudSyncEngine:
    """
    Non-blocking cloud sync orchestrator.
    Manages upload queue, connectivity detection, and retry logic.
    All methods are safe to call even when offline — they degrade gracefully.
    """

    def __init__(self, manifest_path=None):
        self.manifest = UploadManifest(db_path=manifest_path)
        self.uploader = AzureBlobUploader()
        self._is_online = False
        self._last_connectivity_check = None
        self._check_interval = DEFAULT_CHECK_INTERVAL
        self._prune_days = DEFAULT_PRUNE_DAYS
        self._bg_thread = None
        self._stop_event = threading.Event()
        self._upload_lock = threading.Lock()

        print(
            f"[CloudSyncEngine] Initialised. "
            f"Azure SDK: {'yes' if self.uploader.sdk_installed else 'NO'}, "
            f"Azure configured: {'yes' if self.uploader.is_configured else 'NO'}, "
            f"Check interval: {self._check_interval}s, "
            f"Prune retention: {self._prune_days}d"
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    @property
    def status(self):
        """Returns a JSON-serialisable status snapshot for the API."""
        stats = self.manifest.get_stats()
        return {
            "engine_active": self._bg_thread is not None and self._bg_thread.is_alive(),
            "azure_sdk_installed": self.uploader.sdk_installed,
            "azure_configured": self.uploader.is_configured,
            "is_online": self._is_online,
            "container_name": self.uploader.container_name,
            "path_structure": "{container}/{survey_id}/{filename}",
            "max_single_put_size_bytes": self.uploader.MAX_SINGLE_PUT_SIZE,
            "single_put_confirmed_for_reports": True,
            "last_connectivity_check": self._last_connectivity_check,
            "check_interval_seconds": self._check_interval,
            "prune_retention_days": self._prune_days,
            "retry_backoff_schedule": [
                f"{s // 60}min" if s < 3600 else f"{s // 3600}hr"
                for s in RETRY_BACKOFF_SECONDS
            ],
            "manifest_stats": stats,
        }

    # ------------------------------------------------------------------
    # Queue & immediate upload
    # ------------------------------------------------------------------
    def queue_report(self, survey_id, file_path, report_format=None):
        """
        Queues a report for cloud upload (creates manifest entry, status=pending).
        If currently online, fires an immediate background upload attempt.
        """
        try:
            entry = self.manifest.create_entry(survey_id, file_path, report_format)
        except FileNotFoundError:
            return {"error": f"File not found: {file_path}", "queued": False}

        # Fire-and-forget immediate attempt if online
        if self._is_online and entry and entry.get("upload_status") == "pending":
            threading.Thread(
                target=self._upload_entry, args=(entry["id"],), daemon=True
            ).start()

        return {"entry": entry, "queued": True}

    def upload_now(self, entry_id):
        """
        Immediately attempts upload for a specific manifest entry.
        Resets failed_manual entries so they can be retried.
        """
        entry = self.manifest.get_entry(entry_id)
        if not entry:
            return {"success": False, "error": "Entry not found"}
        if entry["upload_status"] == "uploaded":
            return {"success": True, "message": "Already uploaded", "entry": entry}
        if entry["upload_status"] in ("failed", "failed_manual"):
            self.manifest.reset_for_retry(entry_id)

        return self._upload_entry(entry_id)

    def upload_all_as_zip(self, survey_id):
        """
        Creates a ZIP of all pending/failed report files and uploads it.
        On success, marks all included individual entries as uploaded.
        """
        actionable = self.manifest.get_actionable()
        if not actionable:
            return {"success": False, "error": "No pending reports to upload"}

        valid_entries = [e for e in actionable if Path(e["file_path"]).exists()]
        if not valid_entries:
            return {"success": False, "error": "No valid report files found on disk"}

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        zip_name = f"all_reports_{timestamp}.zip"

        try:
            zip_dir = Path(tempfile.gettempdir()) / "marine_debris_cloud_sync"
            zip_dir.mkdir(parents=True, exist_ok=True)
            zip_path = zip_dir / zip_name

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for entry in valid_entries:
                    zf.write(entry["file_path"], entry["file_name"])

            # Queue and upload the ZIP itself
            zip_entry = self.manifest.create_entry(survey_id, str(zip_path), "zip")
            result = self._upload_entry(zip_entry["id"])

            if result.get("success"):
                # Mark all included individual entries as uploaded
                for entry in valid_entries:
                    self.manifest.update_status(
                        entry["id"],
                        "uploaded",
                        azure_blob_url=result.get("blob_url", "uploaded_via_zip"),
                    )

            return result

        except Exception as e:
            return {"success": False, "error": f"ZIP creation failed: {str(e)}"}

    def retry_entry(self, entry_id):
        """Manual retry for a specific entry (resets retry count)."""
        entry = self.manifest.get_entry(entry_id)
        if not entry:
            return {"success": False, "error": "Entry not found"}

        self.manifest.reset_for_retry(entry_id)
        return self._upload_entry(entry_id)

    def retry_all_actionable(self):
        """Retries all pending/retryable entries. Returns list of results."""
        actionable = self.manifest.get_actionable()
        results = []
        for entry in actionable:
            result = self._upload_entry(entry["id"])
            results.append({"entry_id": entry["id"], **result})
        return results

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    def check_connectivity(self):
        """
        Lightweight probe of Azure reachability.
        On connectivity restoration, triggers automatic retry of all actionable entries.
        """
        was_online = self._is_online

        try:
            self._is_online = self.uploader.check_reachability()
        except Exception:
            self._is_online = False

        self._last_connectivity_check = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Connectivity restored → trigger pending retries
        if self._is_online and not was_online:
            print("[CloudSyncEngine] Connectivity restored — retrying pending uploads.")
            threading.Thread(target=self.retry_all_actionable, daemon=True).start()

        return self._is_online

    # ------------------------------------------------------------------
    # Background sync thread
    # ------------------------------------------------------------------
    def start_background_sync(self):
        """Starts the background sync thread (periodic connectivity + retry)."""
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_event.clear()
        self._bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._bg_thread.start()
        print(f"[CloudSyncEngine] Background sync started (interval: {self._check_interval}s)")

    def stop_background_sync(self):
        """Stops the background sync thread gracefully."""
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=5)
        print("[CloudSyncEngine] Background sync stopped.")

    # ------------------------------------------------------------------
    # Manifest queries
    # ------------------------------------------------------------------
    def get_all_entries(self, limit=200):
        return self.manifest.get_all(limit=limit)

    def prune_completed(self):
        """Prunes uploaded tracking rows older than retention period. Files untouched."""
        count = self.manifest.prune_uploaded(retention_days=self._prune_days)
        return {"pruned_count": count, "retention_days": self._prune_days}

    # ------------------------------------------------------------------
    # Internal — single entry upload
    # ------------------------------------------------------------------
    def _upload_entry(self, entry_id):
        """Attempts to upload a single manifest entry to Azure with retry logic."""
        with self._upload_lock:
            entry = self.manifest.get_entry(entry_id)
            if not entry:
                return {"success": False, "error": "Entry not found"}
            if entry["upload_status"] == "uploaded":
                return {"success": True, "message": "Already uploaded"}

            file_path = Path(entry["file_path"])
            if not file_path.exists():
                self.manifest.update_status(
                    entry_id,
                    "failed",
                    error_message=f"Local file missing: {file_path}",
                    increment_retry=True,
                )
                return {"success": False, "error": f"Local file missing: {file_path}"}

            # Mark uploading
            self.manifest.update_status(entry_id, "uploading")

            # Attempt upload
            result = self.uploader.upload_file(
                local_path=str(file_path), survey_id=entry["survey_id"]
            )

            if result.get("success") and result.get("hash_verified", False):
                # ✅ Success — hash verified
                self.manifest.update_status(
                    entry_id,
                    "uploaded",
                    azure_blob_url=result.get("blob_url"),
                    azure_content_md5=result.get("azure_md5_b64"),
                )
                return {"success": True, "blob_url": result.get("blob_url"), "entry_id": entry_id}
            else:
                # ❌ Failed — decide retry vs give-up
                current_retry = entry["retry_count"]
                new_retry = current_retry + 1

                if new_retry >= entry["max_retries"]:
                    # Exceeded max retries → manual retry required
                    self.manifest.update_status(
                        entry_id,
                        "failed_manual",
                        error_message=result.get("error", "Unknown error after max retries"),
                        increment_retry=True,
                    )
                    return {
                        "success": False,
                        "error": result.get("error"),
                        "status": "failed_manual",
                        "message": "Max retries exceeded — manual retry required",
                    }
                else:
                    # Schedule next retry with exponential backoff
                    backoff_idx = min(new_retry, len(RETRY_BACKOFF_SECONDS) - 1)
                    backoff_secs = RETRY_BACKOFF_SECONDS[backoff_idx]
                    next_retry_ts = (
                        datetime.utcnow() + timedelta(seconds=backoff_secs)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")

                    self.manifest.update_status(
                        entry_id,
                        "failed",
                        error_message=result.get("error", "Unknown error"),
                        next_retry_ts=next_retry_ts,
                        increment_retry=True,
                    )
                    return {
                        "success": False,
                        "error": result.get("error"),
                        "retry_count": new_retry,
                        "next_retry": next_retry_ts,
                    }

    # ------------------------------------------------------------------
    # Internal — background loop
    # ------------------------------------------------------------------
    def _background_loop(self):
        """
        Periodic background loop:
        1. On startup: connectivity check + retry actionable + prune
        2. Every check_interval: connectivity check + retry if online
        """
        # Startup actions
        try:
            self.check_connectivity()
            if self._is_online:
                self.retry_all_actionable()
            self.prune_completed()
        except Exception as e:
            print(f"[CloudSyncEngine] Startup sync error: {e}")

        # Periodic loop
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._check_interval)
            if self._stop_event.is_set():
                break
            try:
                self.check_connectivity()
                if self._is_online:
                    actionable = self.manifest.get_actionable()
                    for entry in actionable:
                        if self._stop_event.is_set():
                            break
                        self._upload_entry(entry["id"])
            except Exception as e:
                print(f"[CloudSyncEngine] Background loop error: {e}")
