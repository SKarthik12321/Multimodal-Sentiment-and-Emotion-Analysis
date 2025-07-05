import subprocess
import sys
import platform

def install_ffmpeg():
    print("Starting FFmpeg installation/verification...")

    try:
        # Try to install ffmpeg-python via pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install ffmpeg-python via pip: {e}")
        return False

    # Check if FFmpeg is already installed locally
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("FFmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg is not installed locally. Please install FFmpeg manually from https://ffmpeg.org/download.html")
        print("Instructions for your OS:")
        if platform.system() == "Windows":
            print("Download the Windows build and add it to your PATH.")
        elif platform.system() == "Linux":
            print("Use 'sudo apt-get install ffmpeg' (Ubuntu/Debian) or equivalent for your distribution.")
        elif platform.system() == "Darwin":  # macOS
            print("Use 'brew install ffmpeg' if you have Homebrew installed.")
        return False

if __name__ == "__main__":
    success = install_ffmpeg()
    if not success:
        print("FFmpeg installation/verification failed. Please ensure FFmpeg is installed to proceed.")
        sys.exit(1)