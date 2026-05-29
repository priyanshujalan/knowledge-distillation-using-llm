import datetime
import time

from transformers import AutoTokenizer
import pandas as pd
import torch

from common.csv_helper import read_csv
from common.file_helper import get_path
from common.helper_functions import update_cache
from scripts.sentence_bert import RationaleEncoder
from common.video_helpers import split_modalities_from_video
from models.student.text_bilstm import TextToVectorDataset
from models.student.audio_cnn_bilstm import AudioToVectorDataset
from models.student.video_3dcnn_bilstm import VideoToVectorDataset


def get_torch_dataset(file_path, output_file_prefix):
    # df = read_csv('assets/video_summary.csv')
    df = read_csv(get_path(file_path))
    df = df[(df["number_of_faces"] == 1) & (df["language"].isin(["en", "English"]))]
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

    text_dataset = TextToVectorDataset(all_transcripts, all_text_rationale_vectors, tokenizer)
    audio_dataset = AudioToVectorDataset(all_audio_mels, all_audio_rationale_vectors)
    video_dataset = VideoToVectorDataset(all_video_frames, all_video_rationale_vectors)


    dataset_bundle = {
        'text': text_dataset,
        'audio': audio_dataset,
        'video': video_dataset,
        # We save the parameters to ensure the DataLoaders are identical later
        'loader_params': {
            'batch_size': 2,
            'shuffle': True
        }
    }

    torch_file_name = f'checkpoints/datasets/pt/{output_file_prefix}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.pt'
    # Save to a single file
    torch.save(dataset_bundle, get_path(torch_file_name))
    print(f"Datasets successfully saved to {torch_file_name}")