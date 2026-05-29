import subprocess
from pathlib import Path

from common.constant import MODEL_PROMPT_MAPPER

def get_path(path):
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    return root_dir / path

def get_teacher_prompt(model):
    with open(MODEL_PROMPT_MAPPER[model], 'r') as file:
        content = file.read()

    return content

def get_ffmpeg_version():
    try:
        # Run ffmpeg -version and capture the output
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Grab the first line of the output (e.g., "ffmpeg version 7.0.1...")
        version_line = result.stdout.split('\n')[0]
        print(f"Success: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in the PATH.")
        return False

get_ffmpeg_version()