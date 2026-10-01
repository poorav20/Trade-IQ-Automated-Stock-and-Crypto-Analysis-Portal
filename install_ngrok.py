"""
install_ngrok.py — Helper to download ngrok for Windows automatically.

Run this script once:
    python install_ngrok.py

It will download ngrok.exe into the current folder so launch_mobile.bat
can find it without needing to modify PATH.
"""
import urllib.request
import zipfile
import os
import sys

NGROK_DOWNLOAD_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
DEST_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DEST_DIR, "ngrok.zip")
EXE_PATH = os.path.join(DEST_DIR, "ngrok.exe")


def download_ngrok():
    if os.path.exists(EXE_PATH):
        print(f"[OK] ngrok.exe already exists at: {EXE_PATH}")
        print("     You can delete ngrok.zip if you want.")
        return True

    print("Downloading ngrok for Windows (v3 stable)...")
    print(f"  From: {NGROK_DOWNLOAD_URL}")
    print(f"  To:   {ZIP_PATH}")

    try:
        urllib.request.urlretrieve(NGROK_DOWNLOAD_URL, ZIP_PATH)
        print("Download complete. Extracting...")

        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DEST_DIR)

        if os.path.exists(EXE_PATH):
            print(f"\n[OK] ngrok.exe extracted to: {EXE_PATH}")
            print("\nNext steps:")
            print("  1. Sign up for a free ngrok account at https://ngrok.com/signup")
            print("  2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
            print("  3. Run:  ngrok config add-authtoken YOUR_TOKEN_HERE")
            print("  4. Then run launch_mobile.bat to start the mobile dashboard!")
            return True
        else:
            print("[!] Could not find ngrok.exe after extraction.")
            print("    Contents of zip:")
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                for name in z.namelist():
                    print(f"    - {name}")
            return False

    except Exception as e:
        print(f"[!] Download failed: {e}")
        print("\nManual download:")
        print("  1. Go to: https://ngrok.com/download")
        print("  2. Download the Windows (x86_64) zip")
        print("  3. Extract ngrok.exe into this folder:")
        print(f"     {DEST_DIR}")
        return False


if __name__ == "__main__":
    success = download_ngrok()
    sys.exit(0 if success else 1)
