import os
import urllib.request
import hashlib
import zipfile

from .common import create_folder_if_not_exist

def _check_file_matches_md5(checksum, fpath):
    if not os.path.exists(fpath):
        return False

    with open(fpath, 'rb') as file:
        current_md5checksum = hashlib.md5(file.read()).hexdigest()

    print(f"Expected checksum: {checksum}, Calculated checksum: {current_md5checksum}")
    return current_md5checksum == checksum

def download_monodepth_weight(url, hash, dest_path):
    create_folder_if_not_exist(dest_path)
    model_path = os.path.abspath(dest_path)

    if os.path.exists(os.path.join(model_path, "encoder.pth")):
        print("encoder.pth exists, skipping download.")
    else:
        zip_file_path = f"{model_path}.zip"
        print(f"Zip file path: {zip_file_path}")

        if not _check_file_matches_md5(hash, zip_file_path):
            print('Downloading file...')
            urllib.request.urlretrieve(url, zip_file_path)

        if not _check_file_matches_md5(hash, zip_file_path):
            raise ValueError("Failed to download a file which matches the checksum - quitting")

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)

        try:
            os.remove(zip_file_path)
        except OSError as e:
            raise ValueError(f"Error deleting zip file at {zip_file_path}: {e}")


def verify_sha256(file_path, expected_sha256):
    """
    Verifies the SHA256 hash of a file.

    Args:
        file_path (str): Path to the file to be verified.
        expected_sha256 (str): The expected SHA256 hash.

    Returns:
        bool: True if the file's SHA256 hash matches the expected hash, False otherwise.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_sha256
