"""
Cloud Sync Upload Manifest — SQLite persistence layer.

Tracks upload STATUS only for generated survey reports.
- Survives app restart / device reboot (SQLite WAL-mode file database)
- NEVER deletes local report files — only tracks their upload state
- Prunes completed (uploaded) tracking rows after configurable retention (30 days)
- Is strictly additive/optional — report generation never depends on this module
"""

import os
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Default manifest location — inside pipeline/cloud_sync_data/
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parent.parent.parent / "cloud_sync_data"


class UploadManifest:
    """Thread-safe SQLite manifest for cloud upload tracking."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else (DEFAULT_MANIFEST_DIR / "manifest.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _get_conn(self):
        """Creates a new per-call connection (SQLite is thread-safe in WAL mode)."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        """Creates schema if it doesn't exist."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS upload_manifest (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        survey_id       TEXT    NOT NULL,
                        file_path       TEXT    NOT NULL,
                        file_name       TEXT    NOT NULL,
                        report_format   TEXT,
                        file_size_bytes INTEGER DEFAULT 0,
                        local_hash_sha256 TEXT,
                        upload_status   TEXT    NOT NULL DEFAULT 'pending',
                        retry_count     INTEGER NOT NULL DEFAULT 0,
                        max_retries     INTEGER NOT NULL DEFAULT 5,
                        last_attempt_ts TEXT,
                        next_retry_ts   TEXT,
                        azure_blob_url  TEXT,
                        azure_content_md5 TEXT,
                        error_message   TEXT,
                        created_at      TEXT    NOT NULL,
                        updated_at      TEXT    NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_manifest_status
                        ON upload_manifest(upload_status);
                    CREATE INDEX IF NOT EXISTS idx_manifest_survey
                        ON upload_manifest(survey_id);
                    CREATE INDEX IF NOT EXISTS idx_manifest_next_retry
                        ON upload_manifest(next_retry_ts);

                    CREATE TABLE IF NOT EXISTS manifest_meta (
                        key   TEXT PRIMARY KEY,
                        value TEXT
                    );
                """)
                conn.execute(
                    "INSERT OR REPLACE INTO manifest_meta (key, value) VALUES (?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION)),
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def create_entry(self, survey_id, file_path, report_format=None):
        """
        Creates a new manifest entry for a report file.
        Returns the entry dict, or the existing entry if a pending duplicate exists.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Report file not found: {file_path}")

        file_size = file_path.stat().st_size
        file_hash = self._compute_sha256(file_path)
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._lock:
            conn = self._get_conn()
            try:
                # Deduplicate: same path + same hash already pending/uploading
                existing = conn.execute(
                    """SELECT * FROM upload_manifest
                       WHERE file_path = ? AND local_hash_sha256 = ?
                       AND upload_status IN ('pending', 'uploading')""",
                    (str(file_path), file_hash),
                ).fetchone()
                if existing:
                    return dict(existing)

                cursor = conn.execute(
                    """INSERT INTO upload_manifest
                       (survey_id, file_path, file_name, report_format,
                        file_size_bytes, local_hash_sha256, upload_status,
                        retry_count, max_retries, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, 5, ?, ?)""",
                    (
                        survey_id,
                        str(file_path),
                        file_path.name,
                        report_format,
                        file_size,
                        file_hash,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return self._row_to_dict(conn, cursor.lastrowid)
            finally:
                conn.close()

    def update_status(
        self,
        entry_id,
        status,
        *,
        error_message=None,
        azure_blob_url=None,
        azure_content_md5=None,
        next_retry_ts=None,
        increment_retry=False,
    ):
        """Updates the upload status of a manifest entry."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._lock:
            conn = self._get_conn()
            try:
                retry_clause = ", retry_count = retry_count + 1" if increment_retry else ""
                conn.execute(
                    f"""UPDATE upload_manifest SET
                           upload_status     = ?,
                           last_attempt_ts   = ?,
                           updated_at        = ?,
                           error_message     = COALESCE(?, error_message),
                           azure_blob_url    = COALESCE(?, azure_blob_url),
                           azure_content_md5 = COALESCE(?, azure_content_md5),
                           next_retry_ts     = ?
                           {retry_clause}
                       WHERE id = ?""",
                    (
                        status,
                        now,
                        now,
                        error_message,
                        azure_blob_url,
                        azure_content_md5,
                        next_retry_ts,
                        entry_id,
                    ),
                )
                conn.commit()
                return self._row_to_dict(conn, entry_id)
            finally:
                conn.close()

    def get_entry(self, entry_id):
        """Gets a single manifest entry by ID."""
        conn = self._get_conn()
        try:
            return self._row_to_dict(conn, entry_id)
        finally:
            conn.close()

    def get_all(self, limit=200):
        """Gets all manifest entries, newest first."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM upload_manifest ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_actionable(self):
        """Gets entries that are pending or retryable (due for next retry)."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM upload_manifest
                   WHERE upload_status = 'pending'
                   OR (upload_status = 'failed'
                       AND retry_count < max_retries
                       AND (next_retry_ts IS NULL OR next_retry_ts <= ?))
                   ORDER BY created_at ASC""",
                (now,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_by_status(self, status):
        """Gets entries filtered by upload status."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM upload_manifest WHERE upload_status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def reset_for_retry(self, entry_id):
        """Resets a failed/failed_manual entry for manual retry."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE upload_manifest
                       SET upload_status = 'pending', retry_count = 0,
                           error_message = NULL, next_retry_ts = NULL,
                           updated_at = ?
                       WHERE id = ? AND upload_status IN ('failed', 'failed_manual')""",
                    (now, entry_id),
                )
                conn.commit()
                return self._row_to_dict(conn, entry_id)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def prune_uploaded(self, retention_days=30):
        """
        Removes TRACKING ROWS for uploads completed more than retention_days ago.
        NEVER deletes the actual report files on disk — only the manifest row.
        """
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "DELETE FROM upload_manifest WHERE upload_status = 'uploaded' AND updated_at < ?",
                    (cutoff,),
                )
                conn.commit()
                return result.rowcount
            finally:
                conn.close()

    def get_stats(self):
        """Returns aggregate statistics about the manifest (size estimates, etc.)."""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM upload_manifest").fetchone()[0]

            by_status = {}
            for row in conn.execute(
                "SELECT upload_status, COUNT(*) AS cnt FROM upload_manifest GROUP BY upload_status"
            ).fetchall():
                by_status[row["upload_status"]] = row["cnt"]

            total_bytes = conn.execute(
                "SELECT COALESCE(SUM(file_size_bytes), 0) FROM upload_manifest"
            ).fetchone()[0]

            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_entries": total,
                "by_status": by_status,
                "total_tracked_bytes": total_bytes,
                "manifest_db_size_bytes": db_size,
                "manifest_db_path": str(self.db_path),
                # Size estimates for the operator
                "estimated_1k_entries_kb": round(total_bytes / 1024, 1) if total > 0 else "~25 KB (schema + indices)",
                "estimated_10k_entries_kb": "~250 KB (negligible)",
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _row_to_dict(self, conn, entry_id):
        row = conn.execute("SELECT * FROM upload_manifest WHERE id = ?", (entry_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _compute_sha256(file_path):
        """Computes SHA256 hash of a file (streamed, memory-safe)."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
