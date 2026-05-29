import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
GEMINI_PROJECT_ID = os.environ.get("GEMINI_PROJECT_ID")
GEMINI_REGION = os.environ.get("GEMINI_REGION")

QWEN_API_KEY = os.environ.get("QWEN_API_KEY")
QWEN_MODEL = os.environ.get("QWEN_MODEL")

NEMOTRON_API_KEY = os.environ.get("NEMOTRON_API_KEY")
NEMOTRON_MODEL = os.environ.get("NEMOTRON_MODEL")

KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
KIMI_MODEL = os.environ.get("KIMI_MODEL")

CURRENT_TEACHER_MODEL = os.environ.get("CURRENT_TEACHER_MODEL")

DOWNLOAD_LIMIT_PER_SOURCE = os.environ.get("DOWNLOAD_LIMIT_PER_SOURCE") # 5
BASE_DOWNLOAD_PATH =  os.environ.get("BASE_DOWNLOAD_PATH") # "../Global_6000_Dataset"