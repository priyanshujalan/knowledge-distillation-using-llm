import torch
import torch.nn.functional as F
import numpy as np
import librosa
import cv2
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

# Import your model architecture
from common.constant import EMOTIONS
from scripts.multimodal_emotion_model import MultimodalEmotionModel

class EmotionPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultimodalEmotionModel(num_classes=7).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")

        self.model.eval()
        self.mp_face_mesh = mp.solutions.face_mesh

    def _process_text(self, text, max_len=20):
        """
        Simple hash-based tokenizer to convert text to IDs.
        In production, load the SAME vocab/tokenizer used during training.
        """
        # Simple hashing for demonstration (since we trained on dummy ints)
        # In reality, use: tokenizer.encode(text)
        tokens = [hash(word) % 10000 for word in text.split()]

        # Pad or Truncate to max_len
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return torch.tensor([tokens], dtype=torch.long).to(self.device)

    def _process_audio(self, audio_path, target_len=100):
        """
        Extract MFCCs: (1, 40, 100)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=3.0)  # Limit to 3 secs

            # Extract MFCC (40 channels)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

            # Resize/Pad to fixed time length (target_len)
            current_len = mfcc.shape[1]
            if current_len < target_len:
                pad_width = target_len - current_len
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
            else:
                mfcc = mfcc[:, :target_len]

            tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Add batch dim
            return tensor.to(self.device)

        except Exception as e:
            print(f"Audio processing failed: {e}")
            return torch.zeros(1, 40, target_len).to(self.device)

    def _process_video(self, video_path):
        """
        Extract Face Landmarks from the middle frame of the video.
        Output: (1, 936) flattened vector
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("⚠️ Could not open video.")
            return torch.zeros(1, 936).to(self.device)

        # Read middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return torch.zeros(1, 936).to(self.device)

        # Extract Landmarks using MediaPipe
        with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
        ) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # Flatten x,y coordinates (468 points * 2 = 936)
                # We ignore z for 2D simplicty
                flat_list = []
                for lm in landmarks:
                    flat_list.extend([lm.x, lm.y])

                return torch.tensor([flat_list], dtype=torch.float32).to(self.device)

        print("No face detected in video.")
        return torch.zeros(1, 936).to(self.device)

    def predict(self, video_path, subtitle_text):
        """
        Main function to predict emotion from raw files.
        """
        print(f"Processing Video: {video_path}")

        # 1. Extract Audio from Video
        temp_audio_path = "temp_extracted_audio.wav"
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is not None:
                video_clip.audio.write_audiofile(temp_audio_path, logger=None)
            else:
                print("Video has no audio track.")
        except Exception as e:
            print(f"Error extracting audio: {e}")

        # 2. Preprocess Inputs
        t_tensor = self._process_text(subtitle_text)
        a_tensor = self._process_audio(temp_audio_path) if os.path.exists(temp_audio_path) else torch.zeros(1, 40, 100)
        v_tensor = self._process_video(video_path)

        # 3. Inference
        with torch.no_grad():
            logits = self.model(t_tensor, a_tensor, v_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs).item()

        # Cleanup
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        result = {
            "emotion": EMOTIONS[pred_idx],
            "confidence": probs[0][pred_idx].item(),
            "probabilities": {e: p.item() for e, p in zip(EMOTIONS, probs[0])}
        }

        return result


if __name__ == "__main__":
    # 1. Initialize Predictor
    predictor = EmotionPredictor("checkpoints/models/student_emotion_model.pth")

    # 2. Define Inputs
    video_file = "assets/Video Project Edited.mp4"  # Replace with your actual file path
    subtitle = "We are all done now! Let me just take that away from you!"

    # Note: Ensure 'sample_dialogue.mkv' exists or this will output warnings
    if os.path.exists(video_file):
        prediction = predictor.predict(video_file, subtitle)

        print("\n--- Prediction Result ---")
        print(f"Emotion: {prediction['emotion']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
    else:
        print(f"\nℹTo run this, place a video file named '{video_file}' in the folder.")