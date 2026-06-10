import datetime
import time
import os

from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from common.csv_helper import read_csv
from common.file_helper import get_path
from common.helper_functions import update_cache
from scripts.sentence_bert import RationaleEncoder
from common.video_helpers import split_modalities_from_video
from models.student.text_bilstm import TextToVectorDataset
from models.student.audio_cnn_bilstm import AudioToVectorDataset
from models.student.video_3dcnn_bilstm import VideoToVectorDataset


def get_torch_dataset(file_path, output_file_prefix, existing_torch_file=None):
    # Load the dataframe and filter
    df = read_csv(get_path(file_path))
    df = df[(df["number_of_faces"] == 1) & (df["language"].isin(["en", "English"]))]

    # Check for an existing dataset bundle
    old_datasets = {}
    processed_files = []

    if existing_torch_file and os.path.exists(get_path(existing_torch_file)):
        print(f"Loading existing datasets from {existing_torch_file}...")
        bundle = torch.load(get_path(existing_torch_file))

        old_datasets['text'] = bundle.get('text')
        old_datasets['audio'] = bundle.get('audio')
        old_datasets['video'] = bundle.get('video')

        # Retrieve the ledger of already processed file names
        processed_files = bundle.get('processed_files', [])
    else:
        print("No existing file provided or found. Starting fresh.")

    # 3. Filter the dataframe to ONLY include new files
    if processed_files:
        initial_len = len(df)
        df = df[~df['file_name'].isin(processed_files)]
        print(f"Filtered out {initial_len - len(df)} already processed files. {len(df)} new files to process.")

    # If there are no new rows, we can just return early
    if df.empty:
        print("No new files to process in the dataframe. Exiting.")
        return existing_torch_file

    # 4. Initialize models for the new data
    r_enc = RationaleEncoder()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    df['text_cue'] = df['text_cue'].astype(str)
    df['audio_cue'] = df['audio_cue'].astype(str)
    df['visual_cue'] = df['visual_cue'].astype(str)

    all_transcripts = []
    all_audio_mels = []
    all_video_frames = []

    all_text_rationale_vectors = []
    all_audio_rationale_vectors = []
    all_video_rationale_vectors = []

    cache = {}

    for index, row in df.iterrows():
        start_time = time.time()
        video, audio, transcript = split_modalities_from_video(
            file_name="dataset_videos/"+row['file_name'],
            num_frames_per_second=1
        )

        transcript = transcript.strip()

        all_transcripts.append(transcript)
        all_audio_mels.append(audio)
        all_video_frames.append(video)

        text_rationale_vector = None if pd.isna(row['text_cue']) else update_cache(cache, row['text_cue'], r_enc.get_feature_vector(row['text_cue']))
        audio_rationale_vector = None if pd.isna(row['audio_cue']) else update_cache(cache, row['audio_cue'], r_enc.get_feature_vector(row['audio_cue']))
        visual_rationale_vector = None if pd.isna(row['visual_cue']) else update_cache(cache, row['visual_cue'], r_enc.get_feature_vector(row['visual_cue']))

        all_text_rationale_vectors.append(text_rationale_vector)
        all_audio_rationale_vectors.append(audio_rationale_vector)
        all_video_rationale_vectors.append(visual_rationale_vector)

        print(f"index: {index} Processing Time: {time.time() - start_time}")

    new_text_dataset = TextToVectorDataset(all_transcripts, all_text_rationale_vectors, tokenizer)
    new_audio_dataset = AudioToVectorDataset(all_audio_mels, all_audio_rationale_vectors)
    new_video_dataset = VideoToVectorDataset(all_video_frames, all_video_rationale_vectors)

    # Combine old datasets with new datasets (if old ones exist)
    if old_datasets and old_datasets['text'] is not None:
        final_text_dataset = ConcatDataset([old_datasets['text'], new_text_dataset])
        final_audio_dataset = ConcatDataset([old_datasets['audio'], new_audio_dataset])
        final_video_dataset = ConcatDataset([old_datasets['video'], new_video_dataset])
    else:
        final_text_dataset = new_text_dataset
        final_audio_dataset = new_audio_dataset
        final_video_dataset = new_video_dataset

    # Update our ledger of processed files
    updated_processed_files = processed_files + df['file_name'].tolist()

    # Package the new bundle
    dataset_bundle = {
        'text': final_text_dataset,
        'audio': final_audio_dataset,
        'video': final_video_dataset,
        'processed_files': updated_processed_files,  # <-- New key tracks files
        'loader_params': {
            'batch_size': 2,
            'shuffle': True
        }
    }

    # Save the output
    torch_file_name = f'checkpoints/datasets/pt/{output_file_prefix}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.pt'

    torch.save(dataset_bundle, get_path(torch_file_name))
    print(f"Datasets successfully saved to {torch_file_name}")

    return None
