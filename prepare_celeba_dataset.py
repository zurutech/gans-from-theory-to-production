# Inspired by the public Gist:
# https://gist.github.com/charlesreid1/4f3d676b33b95fce83af08e4ec261822
# Released under public domain license.
"""
Downloads and extracts the Celeb-A dataset in the ./assets folder
"""

import sys
import os
from zipfile import ZipFile
import requests

# Download path and number of bytes to download
# per chunk
URL = "https://docs.google.com/uc?export=download"
CHUNK_SIZE = 32768


def _get_confirm_token(response):
    """Get the confirmation token."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination):
    """Save the response content."""

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def _download_file_from_google_drive(idx, destination):
    """Download the file from google drive."""

    session = requests.Session()

    response = session.get(URL, params={"id": idx}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": idx, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _unzip_file(path_to_file, extract_path):
    """Unzip the file"""
    with ZipFile(path_to_file) as zipfile:
        zipfile.extractall(path=extract_path)


def prepare_celeba_dataset(drive_id, assets_folder):
    """Downlod the Celeb-A dataset, create the asset folder
    and extract the dataset therein."""
    print("Download in progress, please wait.")
    celeba_zip_path = os.path.join(assets_folder, "celeba.zip")
    celeba_folder = os.path.join(assets_folder, "celeba")
    _download_file_from_google_drive(drive_id, celeba_zip_path)
    _unzip_file(celeba_zip_path, assets_folder)
    os.remove(celeba_zip_path)
    os.rename(os.path.join(assets_folder, os.listdir(assets_folder)[0]), celeba_folder)
    print("Done!")


def main():
    """main program"""

    if not os.path.exists("assets"):
        os.makedirs("assets")
    if not os.path.exists("assets/celeba"):
        prepare_celeba_dataset(
            drive_id="0B7EVK8r0v71pZjFTYXZWM3FlRnM", assets_folder="assets"
        )
    else:
        print("CelebA Dataset is already present, bravo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
