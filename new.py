import time

from transformers import AutoTokenizer
import torch

from models.student.text_bilstm import TextToVectorDataset, DataLoader, BiLSTMTextToVector, load_data_and_train_for_text
from models.student.audio_cnn_bilstm import AudioToVectorDataset, CNNBiLSTMAudioToVector, load_data_and_train_for_audio, collate_fn
from models.student.video_3dcnn_bilstm import VideoToVectorDataset, CNN3DBiLSTMVideoToVector, load_data_and_train_for_video, collate_fn_for_video

print("--- Model and Dataset Loading Process ---")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

torch.serialization.add_safe_globals([
    TextToVectorDataset,
    AudioToVectorDataset,
    VideoToVectorDataset
])
all_datasets = torch.load('checkpoints/datasets/pt/multimodal_dataset_2026-03-10-04-30.pt', weights_only=False)


text_dataset = all_datasets['text']
audio_dataset = all_datasets['audio']
video_dataset = all_datasets['video']

text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True)
audio_dataloader = DataLoader(audio_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
video_dataloader = DataLoader(video_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_for_video)

text_model = BiLSTMTextToVector(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    output_dim=384,
    num_layers=2,
    dropout=0.3
)

audio_model = CNNBiLSTMAudioToVector(
    output_dim=384,
    cnn_channels=[32, 64, 128],
    lstm_hidden=256,
    lstm_layers=2,
    dropout=0.3
)

video_model = CNN3DBiLSTMVideoToVector(
    input_channels=3,
    output_dim=384,
    cnn_channels=[64, 128, 256, 512, 512],
    lstm_hidden=512,
    lstm_layers=2,
    dropout=0.3
)

start_time = time.time()
text_trainer = load_data_and_train_for_text(text_model, text_dataloader, num_epochs=10)
print("Text Training Time: ", time.time() - start_time)
start_time = time.time()
audio_trainer = load_data_and_train_for_audio(audio_model, audio_dataloader, num_epochs=10)
print("Audio Training Time: ", time.time() - start_time)
start_time = time.time()
video_trainer = load_data_and_train_for_video(video_model, video_dataloader, num_epochs=10)
print("Video Training Time: ", time.time() - start_time)


# Prediction
# test_text = all_transcripts[10]
# test_audio = all_audio_mels[10]
# test_video = all_video_frames[10]
#
# text_predicted_vector = text_trainer.predict(test_text, tokenizer)
# score = cosine_similarity(text_predicted_vector.reshape(1, -1), all_text_rationale_vectors[10].reshape(1, -1))
# print(f"Text: {score=}")
#
# audio_predicted_vector = audio_trainer.predict(test_audio)
# score = cosine_similarity(audio_predicted_vector.reshape(1, -1), all_audio_rationale_vectors[10].reshape(1, -1))
# print(f"Audio: {score=}")
#
# video_predicted_vector = video_trainer.predict(test_video)
# score = cosine_similarity(video_predicted_vector.reshape(1, -1), all_video_rationale_vectors[10].reshape(1, -1))
# print(f"Video: {score=}")