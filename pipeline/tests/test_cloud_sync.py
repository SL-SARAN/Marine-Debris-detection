"""
Validation test: Cloud sync layer is non-blocking, offline-safe, and Azure-compliant.
Run with:  venv_pipeline\\Scripts\\python.exe pipeline/tests/test_cloud_sync.py
"""
import sys
import os
import tempfile
import hashlib
import base64
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def test_report_generation_without_cloud():
    """Report generation must work even if cloud sync doesn't exist."""
    print("--- Test 1: Report generation works WITHOUT cloud sync ---")
    from pipeline.modules.geotagging.report_export import AnomalyReportEngine
    re = AnomalyReportEngine()
    print("  AnomalyReportEngine init: OK")
    # Build a dummy record
    records = re.build_report_records([{
        "class_name": "ghost_net",
        "calibrated_confidence_pct": 85.0,
        "confidence_tier": "CONFIRMED_ANOMALY",
        "raw_confidence": 0.92,
        "bbox": [10, 20, 100, 150],
        "georeference": {"latitude": 13.0827, "longitude": 80.2707},
        "physics": {"hscr": 0.8, "is_verified": True},
    }])
    assert len(records) == 1
    assert records[0]["classification"] == "ghost_net"
    print(f"  Built {len(records)} report record(s): OK")
    print()


def test_azure_sdk_graceful():
    """Upload must fail gracefully when Azure credentials are not set."""
    print("--- Test 2: Graceful fallback without Azure credentials ---")
    from pipeline.modules.cloud_sync.uploader import AzureBlobUploader, HAS_AZURE_SDK
    print(f"  HAS_AZURE_SDK = {HAS_AZURE_SDK}")
    u = AzureBlobUploader()
    # Test with non-existent file
    r = u.upload_file("dummy_missing.csv", "test-survey")
    assert r["success"] is False
    assert "File not found" in r["error"]
    print(f"  upload_file (missing file): {r['error']}")

    # Test with existing file but no credentials
    test_file = os.path.join("data", "common_sss_dataset", "data.yaml")
    r2 = u.upload_file(test_file, "test-survey")
    assert r2["success"] is False
    print(f"  upload_file (unconfigured): {r2['error'][:60]}...")
    print(f"  check_reachability: {u.check_reachability()}")
    assert u.check_reachability() is False
    print()


def test_container_path_structure():
    """Verify the path structure strictly adheres to {container}/{survey_id}/{filename}."""
    print("--- Test 3: Container Path Structure Verification ---")
    from pipeline.modules.cloud_sync.uploader import AzureBlobUploader
    u = AzureBlobUploader()
    container = u.container_name
    survey_id = "SURVEY-2026-09-26-BAY-OF-BENGAL"
    filename = "anomaly_report_20260926_140000.csv"
    expected_full_path = f"{container}/{survey_id}/{filename}"
    
    # Check property output structure
    test_file = os.path.join("data", "common_sss_dataset", "data.yaml")
    res = u.upload_file(test_file, survey_id, blob_name=filename)
    assert res["container"] == container
    assert res["blob_path"] == f"{survey_id}/{filename}"
    full_path = f"{res['container']}/{res['blob_path']}"
    assert full_path == expected_full_path
    print(f"  Target Container: {container}")
    print(f"  Survey ID:        {survey_id}")
    print(f"  Blob Path:        {res['blob_path']}")
    print(f"  Full Structure:   {full_path}")
    print("  Structure {container}/{survey_id}/{filename}: VERIFIED")
    print()


def test_single_put_behavior_and_threshold_check():
    """
    Verify single-put behavior for KB-size report files and confirm
    that files > 4MB threshold are flagged explicitly to prevent silent hash mismatch.
    """
    print("--- Test 4: Single-Put Behavior & Max Size Threshold Check ---")
    from pipeline.modules.cloud_sync.uploader import AzureBlobUploader
    u = AzureBlobUploader()
    print(f"  MAX_SINGLE_PUT_SIZE threshold: {u.MAX_SINGLE_PUT_SIZE} bytes (4 MB)")

    # 1. Typical report file (small, KB-sized)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"timestamp,survey_id,class,lat,lon\n2026-09-26T12:00:00Z,SURVEY-01,ghost_net,13.08,80.27\n")
        f.flush()
        small_file = f.name

    try:
        small_size = os.path.getsize(small_file)
        print(f"  Report file size: {small_size} bytes (typical KB size)")
        res_small = u.upload_file(small_file, "TEST-SURVEY")
        assert res_small["single_put_confirmed"] is True
        print(f"  single_put_confirmed: {res_small['single_put_confirmed']} (PutBlob REST behavior confirmed)")
    finally:
        if os.path.exists(small_file):
            os.remove(small_file)

    # 2. Simulated oversized file crossing 4MB threshold
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # 4.5 MB = 4.5 * 1024 * 1024 bytes
        oversized_bytes = int(4.5 * 1024 * 1024)
        f.seek(oversized_bytes - 1)
        f.write(b"\0")
        f.flush()
        oversized_file = f.name

    try:
        oversized_size = os.path.getsize(oversized_file)
        print(f"  Oversized file size: {oversized_size} bytes ({oversized_size / 1024 / 1024:.2f} MB > 4MB)")
        res_large = u.upload_file(oversized_file, "TEST-SURVEY")
        assert res_large["success"] is False
        assert res_large["single_put_confirmed"] is False
        assert res_large.get("exceeds_single_put_threshold") is True
        assert "exceeds max single-put threshold" in res_large["error"]
        print(f"  Exceeds threshold flag: {res_large.get('exceeds_single_put_threshold')}")
        print(f"  Flag message: {res_large['error'][:75]}...")
        print("  Threshold enforcement: VERIFIED (chunked block upload flagged, no silent mismatch)")
    finally:
        if os.path.exists(oversized_file):
            os.remove(oversized_file)
    print()


def test_md5_hash_and_content_md5_verification():
    """
    Test MD5 integrity protocol:
    1. Compute local MD5.
    2. Explicitly set Content-MD5 property via ContentSettings.
    3. Verify service-stored Content-MD5 matches local MD5.
    4. Ensure mismatch fails verification and prevents marking status as 'uploaded'.
    """
    print("--- Test 5: MD5 Hash & Content-MD5 Service Verification ---")
    from pipeline.modules.cloud_sync.uploader import AzureBlobUploader
    u = AzureBlobUploader()

    dummy_content = b"anomaly_id,confidence,class_name\n1,0.92,ghost_net\n2,0.88,plastic_debris\n"
    local_md5_raw = hashlib.md5(dummy_content).digest()
    local_md5_b64 = base64.b64encode(local_md5_raw).decode("utf-8")
    print(f"  Local file size: {len(dummy_content)} bytes")
    print(f"  Local MD5 (raw bytes):  {local_md5_raw.hex()}")
    print(f"  Local MD5 (base64):     {local_md5_b64}")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(dummy_content)
        f.flush()
        report_path = f.name

    try:
        # Case A: Successful upload with MATCHING Content-MD5 returned by Azure
        mock_blob_client = MagicMock()
        mock_blob_client.url = "https://testaccount.blob.core.windows.net/marine-debris-reports/S1/report.csv"
        
        # Azure returns properties with ContentSettings containing matching Content-MD5
        mock_props = MagicMock()
        mock_props.content_settings.content_md5 = bytearray(local_md5_raw)
        mock_blob_client.get_blob_properties.return_value = mock_props

        u._connection_string = "DefaultEndpointsProtocol=https;AccountName=dummy;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net"

        with patch.object(u, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_blob_client.return_value = mock_blob_client
            mock_get_client.return_value = mock_client

            res = u.upload_file(report_path, "SURVEY-01", "report.csv")

            # Verify upload_blob was called with explicit content_settings.content_md5
            assert mock_blob_client.upload_blob.called
            call_kwargs = mock_blob_client.upload_blob.call_args[1]
            cs = call_kwargs.get("content_settings")
            assert cs is not None
            assert cs.content_md5 == bytearray(local_md5_raw)
            print("  [Case A: Match]")
            print(f"    upload_blob called with ContentSettings.content_md5: {cs.content_md5.hex()}")
            print(f"    Returned service MD5: {res['azure_md5_b64']}")
            print(f"    Hash verified: {res['hash_verified']}")
            assert res["success"] is True
            assert res["hash_verified"] is True
            assert res["azure_md5_b64"] == local_md5_b64

        # Case B: CORRUPTED / MISMATCHED Content-MD5 returned by service
        corrupted_md5_raw = hashlib.md5(b"corrupted_content").digest()
        mock_props_corrupt = MagicMock()
        mock_props_corrupt.content_settings.content_md5 = bytearray(corrupted_md5_raw)
        mock_blob_client.get_blob_properties.return_value = mock_props_corrupt

        with patch.object(u, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_blob_client.return_value = mock_blob_client
            mock_get_client.return_value = mock_client

            res_corrupt = u.upload_file(report_path, "SURVEY-01", "report.csv")
            print("  [Case B: Mismatch / Corruption]")
            print(f"    Expected MD5: {local_md5_b64}")
            print(f"    Returned MD5: {res_corrupt['azure_md5_b64']}")
            print(f"    Hash verified: {res_corrupt['hash_verified']}")
            print(f"    Error flagged: {res_corrupt['error']}")
            assert res_corrupt["success"] is False
            assert res_corrupt["hash_verified"] is False
            assert "Integrity check failed" in res_corrupt["error"]

    finally:
        if os.path.exists(report_path):
            os.remove(report_path)
    print()


def test_manifest_crud():
    """Manifest CRUD operations."""
    print("--- Test 6: Manifest CRUD ---")
    from pipeline.modules.cloud_sync.manifest import UploadManifest
    tmp = os.path.join(tempfile.gettempdir(), "test_manifest.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    m = UploadManifest(db_path=tmp)

    test_file = os.path.join("data", "common_sss_dataset", "data.yaml")
    entry = m.create_entry("TEST-001", test_file, "yaml")
    print(f"  Created entry: id={entry['id']}, status={entry['upload_status']}")
    assert entry["upload_status"] == "pending"

    # Test deduplication
    entry_dup = m.create_entry("TEST-001", test_file, "yaml")
    assert entry_dup["id"] == entry["id"]
    print("  Deduplication: OK (same ID returned)")

    # Fail + retry_count
    entry2 = m.update_status(entry["id"], "failed", error_message="Test error",
                             increment_retry=True)
    print(f"  After fail: status={entry2['upload_status']}, retry={entry2['retry_count']}")
    assert entry2["retry_count"] == 1

    # Reset for retry
    entry3 = m.reset_for_retry(entry["id"])
    print(f"  After reset: status={entry3['upload_status']}, retry={entry3['retry_count']}")
    assert entry3["upload_status"] == "pending"
    assert entry3["retry_count"] == 0

    # Stats
    stats = m.get_stats()
    print(f"  Stats: {stats['total_entries']} entries, DB size: {stats['manifest_db_size_bytes']} bytes")
    assert stats["total_entries"] >= 1

    # Actionable
    actionable = m.get_actionable()
    assert len(actionable) >= 1
    print(f"  Actionable entries: {len(actionable)}")

    # Cleanup
    os.remove(tmp)
    print()


def test_sync_engine_offline():
    """SyncEngine must initialise and degrade gracefully when offline."""
    print("--- Test 7: SyncEngine graceful offline init & queueing ---")
    from pipeline.modules.cloud_sync.sync_engine import CloudSyncEngine
    tmp = os.path.join(tempfile.gettempdir(), "test_sync.db")
    e = CloudSyncEngine(manifest_path=tmp)
    print(f"  Engine online: {e._is_online}")
    assert e._is_online is False
    status = e.status
    print(f"  Status keys: {list(status.keys())}")
    assert "is_online" in status
    assert "container_name" in status
    assert "path_structure" in status
    assert "max_single_put_size_bytes" in status
    assert "single_put_confirmed_for_reports" in status
    print(f"  Container: {status['container_name']}")
    print(f"  Path structure: {status['path_structure']}")
    print(f"  Max single-put size: {status['max_single_put_size_bytes']} bytes")
    print(f"  Single-put confirmed for reports: {status['single_put_confirmed_for_reports']}")

    # Queue should work even offline
    test_file = os.path.join("data", "common_sss_dataset", "data.yaml")
    result = e.queue_report("OFFLINE-SURVEY", test_file, "yaml")
    assert result["queued"] is True
    print(f"  Queue while offline: queued={result['queued']}")

    entries = e.get_all_entries()
    print(f"  Entries after queue: {len(entries)}")
    assert len(entries) >= 1

    # Cleanup
    if os.path.exists(tmp):
        os.remove(tmp)
    print()


if __name__ == "__main__":
    test_report_generation_without_cloud()
    test_azure_sdk_graceful()
    test_container_path_structure()
    test_single_put_behavior_and_threshold_check()
    test_md5_hash_and_content_md5_verification()
    test_manifest_crud()
    test_sync_engine_offline()
    print("=" * 65)
    print("  ALL TESTS PASSED")
    print("  1. Report generation & detection are fully offline-safe.")
    print("  2. Container structure {container}/{survey_id}/{filename} verified.")
    print("  3. Single-put behavior confirmed for KB-size reports.")
    print("  4. Max size threshold (>4MB) properly flagged.")
    print("  5. Local MD5 vs Azure Content-MD5 verification verified.")
    print("=" * 65)
