import json
import time

from google import genai

from common.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_PROJECT_ID, GEMINI_REGION
from common.dictionary_helper import clean_teacher_response
from common.file_helper import get_teacher_prompt, get_path

CLIENT = genai.Client(
    # api_key=GEMINI_API_KEY,
    vertexai=True,           # This is the toggle to enable regional control
    project=GEMINI_PROJECT_ID,
    location=GEMINI_REGION   # Try 'us-east4' or 'europe-west4' if us-central1 is busy
)

def get_response(video_path):
    my_file = CLIENT.files.upload(file=video_path)
    upload_try = 0
    while my_file.state == 'PROCESSING':
        time.sleep(2)
        upload_try += 1
        if upload_try > 5:
            print("Upload failed. Skipping...")
            return {}
        my_file = CLIENT.files.get(name=my_file.name)

    prompt = get_teacher_prompt(GEMINI_MODEL)
    response = CLIENT.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            my_file,
            prompt
        ],
        config=genai.types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            max_output_tokens=500
        )
    )

    return clean_teacher_response(response.parts[0].text)

if __name__ == "__main__":
    print(get_response(get_path("dataset_videos/4jhuu13ctkc-0.mp4")))