import os
import urllib.request
import hashlib
import requests
from tqdm import tqdm

def download_file(url, dest_path, expected_sha256=None):
    """
    Downloads a file from the given URL and saves it to the specified destination path.
    If the file already exists and its hash matches the expected SHA256, the download is skipped.

    Args:
        url (str): URL to download the file from.
        dest_path (str): Destination path to save the downloaded file.
        expected_sha256 (str, optional): The expected SHA256 hash of the file for verification.
    """
    # Check if the file already exists
    if os.path.exists(dest_path):
        if expected_sha256 and not verify_sha256(dest_path, expected_sha256):
            print(f"Existing file {dest_path} is corrupted, re-downloading...")
        else:
            print(f"File {dest_path} already exists and is verified. Skipping download.")
            return
    else:
        print(f"Downloading file from {url} to {dest_path}...")

    # Download the file with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=4096):
                file.write(data)
                bar.update(len(data))

        # Verify the downloaded file's SHA256 hash
        if expected_sha256 and not verify_sha256(dest_path, expected_sha256):
            raise ValueError("Downloaded file is corrupted (SHA256 mismatch).")
        print("Download completed successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

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
