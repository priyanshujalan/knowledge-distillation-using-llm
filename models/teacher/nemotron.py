import requests
import os
import base64
import sys

from common.config import NEMOTRON_API_KEY, NEMOTRON_MODEL
from common.dictionary_helper import clean_teacher_response
from common.file_helper import get_path, get_teacher_prompt

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

# ext: {mime, media_type}
kSupportedList = {
    "png": ["image/png", "image_url"],
    "jpg": ["image/jpeg", "image_url"],
    "jpeg": ["image/jpeg", "image_url"],
    "webp": ["image/webp", "image_url"],
    "mp4": ["video/mp4", "video_url"],
    "webm": ["video/webm", "video_url"],
    "mov": ["video/mov", "video_url"]
}


def get_extension(filename):
    _, ext = os.path.splitext(filename)
    ext = ext[1:].lower()
    return ext


def mime_type(ext):
    return kSupportedList[ext][0]


def media_type(ext):
    return kSupportedList[ext][1]


def encode_media_base64(media_file):
    """Encode media file to base64 string"""
    with open(media_file, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_response(video_path):

    prompt = get_teacher_prompt(NEMOTRON_MODEL)

    # Build content array with text and media
    content = [{"type": "text", "text": prompt}]

    base64_data = encode_media_base64(video_path)

    ext = get_extension(video_path)
    media_type_key = media_type(ext)

    media_obj = {
        "type": media_type_key,
        media_type_key: {
            "url": f"data:{mime_type(ext)};base64,{base64_data}"
        }
    }
    content.append(media_obj)

    headers = {
        "Authorization": f"Bearer {NEMOTRON_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Add system message with appropriate prompt
    # Videos only support /no_think, images support both

    system_prompt = "/no_think"

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": content,
        }
    ]
    payload = {
        "max_tokens": 4096,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "messages": messages,
        "stream": False,
        "model": f"nvidia/{NEMOTRON_MODEL}",
    }

    response = requests.post(invoke_url, headers=headers, json=payload, stream=False)
    return response.json()


if __name__ == "__main__":
    """ Usage:
        python test.py                                    # Text-only
        python test.py sample.mp4                         # Single video
        python test.py sample1.png sample2.png            # Multiple images
    """

    media_samples = list(sys.argv[1:])
    get_response(media_samples)
