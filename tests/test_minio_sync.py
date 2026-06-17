import pytest
from unittest.mock import MagicMock, patch, call


@pytest.fixture
def minio_mock():
    mock = MagicMock()
    mock.bucket_exists.return_value = True
    mock.list_objects.return_value = []
    return mock


def test_download_folder_creates_local_dir(minio_mock, tmp_path):
    target = tmp_path / "chroma"
    with patch("retie_agent.services.storage_minio._client", return_value=minio_mock):
        from retie_agent.services.storage_minio import download_folder
        download_folder(bucket="test-bucket", prefix="chroma_db/", local_dir=str(target))
    assert target.exists()


def test_upload_folder_calls_fput_per_file(minio_mock, tmp_path):
    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.txt").write_text("world")
    with patch("retie_agent.services.storage_minio._client", return_value=minio_mock):
        from retie_agent.services.storage_minio import upload_folder
        upload_folder(local_dir=str(tmp_path), bucket="test-bucket", prefix="chroma_db")
    assert minio_mock.fput_object.call_count == 2


def test_upload_folder_uses_correct_bucket_and_prefix(minio_mock, tmp_path):
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    with patch("retie_agent.services.storage_minio._client", return_value=minio_mock):
        from retie_agent.services.storage_minio import upload_folder
        upload_folder(local_dir=str(tmp_path), bucket="my-bucket", prefix="models/")
    args = minio_mock.fput_object.call_args[0]
    assert args[0] == "my-bucket"
    assert args[1].startswith("models/")


def test_list_objects_skips_directory_entries(minio_mock):
    dir_obj = MagicMock()
    dir_obj.object_name = "prefix/"
    file_obj = MagicMock()
    file_obj.object_name = "prefix/chroma.sqlite3"
    minio_mock.list_objects.return_value = [dir_obj, file_obj]
    with patch("retie_agent.services.storage_minio._client", return_value=minio_mock):
        from retie_agent.services.storage_minio import list_objects
        result = list(list_objects(bucket="b", prefix="prefix/"))
    assert result == ["prefix/chroma.sqlite3"]


def test_upload_empty_folder_makes_no_calls(minio_mock, tmp_path):
    with patch("retie_agent.services.storage_minio._client", return_value=minio_mock):
        from retie_agent.services.storage_minio import upload_folder
        upload_folder(local_dir=str(tmp_path), bucket="bucket", prefix="pref/")
    minio_mock.fput_object.assert_not_called()
