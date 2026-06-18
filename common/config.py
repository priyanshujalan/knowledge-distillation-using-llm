import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
GEMINI_PROJECT_ID = os.environ.get("GEMINI_PROJECT_ID")
GEMINI_REGION = os.environ.get("GEMINI_REGION")

QWEN_3_5_122B_A10B_API_KEY = os.environ.get("QWEN3.5_122B_A10B_API_KEY")
NEMOTRON_NANO_12B_V2_VL_API_KEY = os.environ.get("NEMOTRON_NANO_12B_V2_VL_API_KEY")
KIMI_K2_6_API_KEY = os.environ.get("KIMI_K2.6_API_KEY")
GEMMA4_31B_IT_API_KEY = os.environ.get("GEMMA4_31B_IT_API_KEY")
DIFUSIONGEMMA_26B_A4B_IT_API_KEY = os.environ.get("DIFUSIONGEMMA_26B_A4B_IT_API_KEY")

CURRENT_TEACHER_MODEL = os.environ.get("CURRENT_TEACHER_MODEL")

DOWNLOAD_LIMIT_PER_SOURCE = os.environ.get("DOWNLOAD_LIMIT_PER_SOURCE") # 5
BASE_DOWNLOAD_PATH =  os.environ.get("BASE_DOWNLOAD_PATH") # "../Global_6000_Dataset"


def get_api_key(model=CURRENT_TEACHER_MODEL):
    api_key = None
    if model == 'gemini':
        api_key = GEMINI_API_KEY
    elif model == 'gemma-4-31b-it':
        api_key = GEMMA4_31B_IT_API_KEY
    elif model == 'nemotron-nano-12b-v2-vl':
        api_key = NEMOTRON_NANO_12B_V2_VL_API_KEY
    elif model == 'qwen3.5-122b-a10b':
        api_key = QWEN_3_5_122B_A10B_API_KEY
    elif model == 'diffusiongemma-26b-a4b-it':
        api_key = DIFUSIONGEMMA_26B_A4B_IT_API_KEY
    elif model == 'kimi-k2.6':
        api_key = KIMI_K2_6_API_KEY
    
    return api_key, CURRENT_TEACHER_MODEL