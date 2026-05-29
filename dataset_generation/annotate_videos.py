import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import keyboard
import pandas as pd

from common.config import CURRENT_TEACHER_MODEL
from common.constant import EMOTIONS
from models.teacher.inference import get_response


MAX_WORKERS = 5


def process_file(path_name, file_name):
    """Worker function to handle a single file request."""
    try:
        # Check for quit signal (Note: keyboard check is tricky inside threads,
        # usually better handled in the main loop, but we'll keep the logic here)
        print(f"Analyzing: {file_name}...")

        response = get_response(CURRENT_TEACHER_MODEL, f"{path_name}\\{file_name}")

        if response and response != {}:
            return file_name, response
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

    return file_name, None

def annotate_downloaded_videos(videos_folder_path, db_file_path, csv_output):  # ("../dataset_videos", "../first_set_of_annotated_data.json", "annotated_data.csv")
    with open(db_file_path, "r") as hashtags_file:
        db = json.load(hashtags_file)

    path_name = videos_folder_path
    downloaded_mp4_files = [file_name for file_name in os.listdir(path_name) if file_name.endswith(".mp4")]
    unprocessed_mp4_files = list(set(downloaded_mp4_files) - set(db.keys()))

    print(f"Skipping {len(downloaded_mp4_files) - len(unprocessed_mp4_files)} files...")

    # unprocessed_mp4_files = unprocessed_mp4_files[0:5]

    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map all files to the executor
        future_to_file = {executor.submit(process_file, path_name, f): f for f in unprocessed_mp4_files}

        for future in as_completed(future_to_file):
            file_name, teacher_response = future.result()

            if teacher_response:
                results[file_name] = teacher_response
                print(f"Completed: {file_name} -> {teacher_response.get('emotion')}")

    # Update main database
    db.update(results)

    with open(db_file_path, "w") as f:
        json.dump(db, f)
    print(f"Saved the responses to {db_file_path}")

    # Create DataFrame
    db_list = [{"file_name": name, **data} for name, data in db.items()]
    df = pd.DataFrame(db_list)

    if not df.empty and 'emotion' in df.columns:
        df[EMOTIONS] = pd.DataFrame(df['emotion'].tolist(), index=df.index)
        df = df.drop(columns=['emotion'])
        df.to_csv(csv_output, index=False)


if __name__ == "__main__":
    annotate_downloaded_videos("../dataset_videos", "../checkpoints/datasets/json/first_set_of_annotated_data.json", "../annotated_data.csv")


