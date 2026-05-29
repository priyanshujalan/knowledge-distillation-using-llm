import base64
import json
import requests
import time

from common.config import KIMI_API_KEY, KIMI_MODEL
from common.dictionary_helper import clean_teacher_response
from common.file_helper import get_path, get_teacher_prompt


invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

def read_b64(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

def get_response(video_path):
    video_b64s = read_b64(video_path)
    prompt = get_teacher_prompt(KIMI_MODEL)

    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
        "model": f"moonshotai/{KIMI_MODEL}",
        "messages": [
            {
                "role": "user",
                "content": [{"type":"video_url","video_url":{"url":f"data:video/mp4;base64,{video_b64s}"}},{"type":"text","text":prompt}]
            }
        ],
        "max_tokens": 16384,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking":True},
    }

    response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
    # json.loads(response.json()['choices'][0]['message']['content'].replace('```json\n', '').replace('```', ''))
    return clean_teacher_response(response.json()['choices'][0]['message']['content'])

if __name__ == "__main__":
    start_time = time.time()
    print(f"{start_time=}")
    print(get_response(get_path("dataset_videos/4jhuu13ctkc-0.mp4")))
    print(f"Elapsed Time: {time.time() - start_time}")
