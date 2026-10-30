"""
Azure Blob Storage Uploader with integrity verification.

Uploads report files to Azure Blob Storage with:
- SHA256 local hash computation (stored in manifest)
- MD5 content hash verification against Azure's returned Content-MD5 on PutBlob
- Connection string / SAS token / credentials from environment variables only
- Graceful degradation if azure-storage-blob SDK is not installed

Environment variables (never hardcoded, never exposed to frontend):
    AZURE_STORAGE_CONNECTION_STRING   — Full connection string (preferred)
    AZURE_STORAGE_ACCOUNT_NAME        — Storage account name (with SAS token)
    AZURE_STORAGE_SAS_TOKEN           — SAS token for account-level access
    AZURE_STORAGE_CONTAINER_NAME      — Target container (default: marine-debris-reports)
"""

import os
import json
import hashlib
import base64
from pathlib import Path

# Lazy import of Azure SDK — the sync layer must be optional
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings

    HAS_AZURE_SDK = True
except ImportError:
    HAS_AZURE_SDK = False


# Default threshold where Azure SDK switches from PutBlob to StageBlock chunked uploads
DEFAULT_MAX_SINGLE_PUT_SIZE = 4 * 1024 * 1024  # 4 MB


class AzureBlobUploader:
    """Handles Azure Blob Storage uploads with strict hash verification.
    
    Path structure: {container}/{survey_id}/{filename}
    """

    MAX_SINGLE_PUT_SIZE = DEFAULT_MAX_SINGLE_PUT_SIZE

    def __init__(self, secrets_path=None):
        self._secrets_path = secrets_path or os.environ.get("AZURE_SECRETS_FILE", "")
        self._connection_string = ""
        self._sas_token = ""
        self._account_name = ""
        self._container_name = "marine-debris-reports"
        self._client = None

        self._load_credentials()

    def _load_credentials(self):
        """Loads credentials from environment variables or a local secrets file.
        Credentials are never hardcoded and never exposed to the frontend."""
        # 1. Check environment variables
        self._connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        self._sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN", "")
        self._account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
        self._container_name = os.environ.get(
            "AZURE_STORAGE_CONTAINER_NAME", "marine-debris-reports"
        )

        # 2. Check optional secrets file if env vars not fully set
        if not self.is_configured:
            candidate_files = []
            if self._secrets_path:
                candidate_files.append(Path(self._secrets_path))
            # Standard non-committed locations
            candidate_files.extend([
                Path(".azure_secrets.json"),
                Path("config/azure_secrets.json"),
                Path("pipeline/config/azure_secrets.json"),
            ])
            for fpath in candidate_files:
                if fpath.is_file():
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self._connection_string = data.get("AZURE_STORAGE_CONNECTION_STRING", self._connection_string)
                            self._sas_token = data.get("AZURE_STORAGE_SAS_TOKEN", self._sas_token)
                            self._account_name = data.get("AZURE_STORAGE_ACCOUNT_NAME", self._account_name)
                            self._container_name = data.get("AZURE_STORAGE_CONTAINER_NAME", self._container_name)
                            if self.is_configured:
                                break
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Configuration status
    # ------------------------------------------------------------------
    @property
    def container_name(self):
        return self._container_name

    @property
    def is_configured(self):
        """True if Azure credentials are set via env vars or secrets file."""
        return bool(self._connection_string or (self._account_name and self._sas_token))

    @property
    def is_available(self):
        """True if the Azure SDK is installed AND credentials are configured."""
        return HAS_AZURE_SDK and self.is_configured

    @property
    def sdk_installed(self):
        return HAS_AZURE_SDK

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------
    def _get_client(self):
        """Lazily creates the BlobServiceClient."""
        if self._client is None and self.is_available:
            if self._connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self._connection_string
                )
            elif self._account_name and self._sas_token:
                account_url = f"https://{self._account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url, credential=self._sas_token
                )
        return self._client

    def _ensure_container(self):
        """Creates the container if it doesn't exist (best-effort)."""
        client = self._get_client()
        if client:
            try:
                container_client = client.get_container_client(self._container_name)
                if not container_client.exists():
                    container_client.create_container()
            except Exception:
                pass  # May already exist or ACL may disallow creation

    # ------------------------------------------------------------------
    # Upload with hash verification
    # ------------------------------------------------------------------
    def upload_file(self, local_path, survey_id, blob_name=None):
        """
        Uploads a single file to Azure Blob Storage with integrity verification.

        Path structure: {container}/{survey_id}/{filename}

        Integrity Protocol:
        1. Confirms file size <= max_single_put_size (4MB) to ensure single-put
           behavior ('Put Blob' REST call).
        2. If file size > 4MB, flags this explicitly rather than silently mismatching
           chunked block upload hashes.
        3. Computes local MD5 hash before upload.
        4. Explicitly sets Content-MD5 property in ContentSettings on upload_blob.
        5. Validates Content-MD5 returned by the service properties matches local MD5.
        6. Only returns success=True when hashes match.

        Returns dict with:
            success, blob_url, blob_path, container, single_put_confirmed,
            local_sha256, local_md5_b64, azure_md5_b64, hash_verified,
            file_size_bytes, error
        """
        local_path = Path(local_path)
        if not local_path.exists():
            return {
                "success": False,
                "error": f"File not found: {local_path}",
                "single_put_confirmed": False,
                "hash_verified": False,
            }

        file_size = local_path.stat().st_size
        blob_name = blob_name or local_path.name
        blob_path = f"{survey_id}/{blob_name}"

        # Size check: Confirm single-put behavior
        if file_size > self.MAX_SINGLE_PUT_SIZE:
            return {
                "success": False,
                "blob_path": blob_path,
                "container": self._container_name,
                "file_size_bytes": file_size,
                "single_put_confirmed": False,
                "exceeds_single_put_threshold": True,
                "hash_verified": False,
                "error": (
                    f"File size ({file_size} bytes / {file_size / 1024 / 1024:.2f} MB) exceeds "
                    f"max single-put threshold ({self.MAX_SINGLE_PUT_SIZE} bytes / 4 MB). "
                    "Azure SDK switches to chunked block upload (StageBlock/CommitBlockList), "
                    "which prevents single-blob Content-MD5 verification. Flagged to prevent "
                    "silent hash mismatch."
                ),
            }

        single_put_confirmed = True

        if not self.is_available:
            reasons = []
            if not HAS_AZURE_SDK:
                reasons.append("azure-storage-blob package not installed (pip install azure-storage-blob)")
            if not self.is_configured:
                reasons.append("Azure credentials not configured (set AZURE_STORAGE_CONNECTION_STRING or AZURE_SECRETS_FILE)")
            return {
                "success": False,
                "blob_path": blob_path,
                "container": self._container_name,
                "single_put_confirmed": single_put_confirmed,
                "hash_verified": False,
                "error": "; ".join(reasons),
            }

        try:
            # Read file and compute local hashes
            file_data = local_path.read_bytes()
            local_sha256 = hashlib.sha256(file_data).hexdigest()
            local_md5_raw = hashlib.md5(file_data).digest()
            local_md5_b64 = base64.b64encode(local_md5_raw).decode("utf-8")

            # Upload with content MD5 explicitly set in ContentSettings
            client = self._get_client()
            self._ensure_container()
            blob_client = client.get_blob_client(
                container=self._container_name, blob=blob_path
            )

            # Explicitly set Content-MD5 property for Azure Blob Storage
            content_settings = ContentSettings(
                content_type=self._guess_content_type(local_path.suffix),
                content_md5=bytearray(local_md5_raw),
            )

            blob_client.upload_blob(
                file_data,
                overwrite=True,
                content_settings=content_settings,
                max_single_put_size=self.MAX_SINGLE_PUT_SIZE,
                validate_content=True,  # SDK wire-level MD5 verification
            )

            # Post-upload integrity check: read back service-stored Content-MD5
            blob_props = blob_client.get_blob_properties()
            azure_md5_raw = blob_props.content_settings.content_md5

            hash_verified = False
            azure_md5_b64 = ""
            if azure_md5_raw:
                azure_md5_b64 = base64.b64encode(bytes(azure_md5_raw)).decode("utf-8")
                hash_verified = (bytes(azure_md5_raw) == local_md5_raw)
            else:
                # If service did not store Content-MD5 despite content_settings,
                # do not mark as verified
                hash_verified = False
                azure_md5_b64 = "none_returned"

            blob_url = blob_client.url

            return {
                "success": hash_verified,
                "blob_url": blob_url,
                "blob_path": blob_path,
                "container": self._container_name,
                "full_blob_path": f"{self._container_name}/{blob_path}",
                "single_put_confirmed": single_put_confirmed,
                "local_sha256": local_sha256,
                "local_md5_b64": local_md5_b64,
                "azure_md5_b64": azure_md5_b64,
                "hash_verified": hash_verified,
                "file_size_bytes": len(file_data),
                "error": (
                    None
                    if hash_verified
                    else f"Integrity check failed: local_md5={local_md5_b64} vs azure_md5={azure_md5_b64}"
                ),
            }

        except Exception as e:
            return {
                "success": False,
                "blob_path": blob_path,
                "container": self._container_name,
                "single_put_confirmed": single_put_confirmed,
                "hash_verified": False,
                "error": f"Upload failed: {type(e).__name__}: {str(e)}",
            }

    # ------------------------------------------------------------------
    # Connectivity probe
    # ------------------------------------------------------------------
    def check_reachability(self):
        """
        Lightweight probe to check if Azure Blob service is reachable.
        Uses a minimal list_containers(results_per_page=1) call.
        """
        if not self.is_available:
            return False
        try:
            client = self._get_client()
            if client:
                next(client.list_containers(results_per_page=1), None)
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _guess_content_type(suffix):
        """Maps file extension to MIME type."""
        mapping = {
            ".csv": "text/csv",
            ".json": "application/json",
            ".geojson": "application/geo+json",
            ".pdf": "application/pdf",
            ".zip": "application/zip",
        }
        return mapping.get(suffix.lower(), "application/octet-stream")
