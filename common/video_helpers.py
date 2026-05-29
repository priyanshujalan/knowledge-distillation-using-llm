import cv2
import os
import math

import librosa
import numpy as np
from moviepy import VideoFileClip
import mediapipe
from scenedetect import detect, ContentDetector
import whisper

from common.constant import EMOTIONS_VECTOR_MAP
from common.csv_helper import read_csv
from common.helper_functions import coalesce

transcription_model = whisper.load_model("tiny")
face_detection_model = mediapipe.solutions.face_detection
face_detector = face_detection_model.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def split_video(video_path, root_path_for_output='dataset_videos', seperator='-'):

    # ContentDetector uses frame differences to find cuts. Adjust threshold if detection is too sensitive/insensitive (e.g., threshold=30).
    scene_list = detect(video_path, ContentDetector(threshold=27))  # Default threshold is 27; lower for more sensitivity.
    # Load the video and create subclips
    video = VideoFileClip(video_path)

    # Extract start and end times for each scene (in seconds)
    timestamps = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    if len(timestamps) == 0:
        timestamps.append((0, video.duration))
    timestamps[-1] = (timestamps[-1][0], timestamps[-1][1] - 0.01)

    print(f"Detected {len(timestamps)} scenes with timestamps: {timestamps}")

    clips = []
    for start_time, end_time in timestamps:
        clip = video.subclipped(start_time, end_time)
        clips.append(clip)

    video_name = video_path.split('.mp4')[0].split('\\')[-1]
    # Save each clip
    for i, clip in enumerate(clips):
        output_path = f'{root_path_for_output}\\{video_name}{seperator}{i}.mp4'
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print(f"Saved {output_path}")

    # Close the video to free resources
    video.close()



def split_modalities_from_video(file_name, num_frames_per_second=None, resize=(512,512)):

    # Reading video file through moviepy for audio extraction
    video = VideoFileClip(file_name)
    audio = video.audio
    clip_duration = video.duration

    if audio is None:
        print(f"No audio found in {row['File']}")
        return video, None, None

    # Dumping to a temp file for transcription
    temp_filename = "../temp_audio.wav"
    audio.write_audiofile(temp_filename, codec='pcm_s16le', logger=None)

    # Transcription using whisper-openai
    result = transcription_model.transcribe(temp_filename)
    transcript = result['text']

    # Audio converting Mel Spectrogram
    y, sr = librosa.load(temp_filename, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Extracting video frames
    frames = []
    num_frames = math.ceil(clip_duration * num_frames_per_second)


    times = np.linspace(0, video.duration - 0.01, num=num_frames)

    for t in times:

        frame = video.get_frame(t)
        h, w, _ = frame.shape
        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box

            # 1. Convert normalized to pixel units
            bw, bh = bbox.width * w, bbox.height * h
            cx, cy = (bbox.xmin * w) + (bw / 2), (bbox.ymin * h) + (bh / 2)

            # 2. Determine the side length of the "closest square"
            # We add a 'padding' factor (e.g., 1.5) so the face isn't
            # cramped right against the edges of the 512x512 box.
            padding = 1.6
            side = max(bw, bh) * padding

            # 3. Calculate crop boundaries with clamping
            half_s = side / 2
            x1 = int(max(0, cx - half_s))
            y1 = int(max(0, cy - half_s))
            x2 = int(min(w, cx + half_s))
            y2 = int(min(h, cy + half_s))

            # 4. Extract the square-ish crop
            raw_crop = frame[y1:y2, x1:x2]
            # 5. Reshape (Resize) to exactly 512x512
            # INTER_AREA is best for shrinking; INTER_CUBIC is best for zooming
            final_frame = cv2.resize(raw_crop, resize, interpolation=cv2.INTER_AREA)

            if False:
                cv2.imshow("Face Crop Preview", final_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop preview
                    break

        else:
            # Fallback if no face: center crop or black frame
            final_frame = cv2.resize(frame, resize)  # Simple stretch fallback

        frames.append(final_frame)

    video.close()

    return np.array(frames), mel_spectrogram, transcript


if __name__ == "__main__":
    df = read_csv('../assets/video_summary.csv')

    csv_emotions = set(list(df["Emotion"]))
    registered_emotions = set(EMOTIONS_VECTOR_MAP.keys())

    if not csv_emotions.issubset(registered_emotions):
        print("Not Present Emotions: ", csv_emotions - registered_emotions)
    else:
        print("All Emotions Present: ", csv_emotions)


    for index, row in df.iterrows():
        split_modalities_from_video("assets/"+row['File'])