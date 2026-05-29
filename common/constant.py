FLOAT_NAN = float('nan')

EMOTIONS_MAP = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Joy": 3,
    "Neutral": 4,
    "Sadness": 5,
    "Surprise": 6
}

EMOTIONS_VECTOR_MAP = {
    "Excitement": [0, 0, 0, 0.7, 0, 0, 0.3],
    "Anxiety": [0, 0, 0.6, 0, 0.4, 0, 0],
    "Cheerful": [0, 0, 0, 1, 0, 0, 0],
    "Sadness": [0, 0, 0, 0, 0, 1, 0],
    "Guilt": [0, 0.2, 0.3, 0, 0, 0.5, 0],
    "Frustration": [0.6, 0.4, 0, 0, 0, 0, 0],
    "Astonishment": [0, 0, 0, 0, 0.2, 0, 0.8],
    "Exuberance": [0, 0, 0, 0.8, 0, 0, 0.2]
}

MODEL_PROMPT_MAPPER = {
    "gemini-2.0-flash": "prompt_texts/gemini-2_0-flash.txt",
    "qwen3.5-122b-a10b": "prompt_texts/gemini-2_0-flash.txt",
    "nemotron-nano-12b-v2-vl": "prompt_texts/gemini-2_0-flash.txt",
    "kimi-k2.6": "prompt_texts/gemini-2_0-flash.txt"
}

EMOTIONS = list(EMOTIONS_MAP.keys())

for emotion in EMOTIONS_VECTOR_MAP.keys():
    if sum(EMOTIONS_VECTOR_MAP[emotion]) != 1:
        print(emotion, ": Sum is not equal to 1")
