import random

from common.constant import EMOTIONS_MAP, EMOTIONS


def query_llm_teacher(text_content, audio_path, video_path):
    """
    Simulates querying a SOTA LLM (e.g., Gemini 1.5 Pro, GPT-4o).
    """

    # 1. Pick a winner
    winner_idx = random.randint(0, len(EMOTIONS) - 1)
    predicted_emotion = EMOTIONS[winner_idx]

    # 2. Generate high confidence for the winner (0.80 to 0.99)
    high_confidence = random.uniform(0.80, 0.99)

    # 3. Distribute the remaining probability among others
    remaining_prob = 1.0 - high_confidence
    others_count = len(EMOTIONS) - 1

    soft_labels = [0.0] * len(EMOTIONS)
    soft_labels[winner_idx] = high_confidence

    current_sum = 0
    for i in range(len(EMOTIONS)):
        if i != winner_idx:
            share = remaining_prob / others_count
            noise = random.uniform(-share / 2, share / 2)
            val = max(0.0, share + noise)
            soft_labels[i] = val
            current_sum += val

    total_prob = sum(soft_labels)
    soft_labels = [x / total_prob for x in soft_labels]
    max_conf = max(soft_labels)

    return {
        "emotion": predicted_emotion,
        "confidence": max_conf,
        "soft_labels": soft_labels,
        "class_order": EMOTIONS
    }