# Multimodal Emotion Recognition — Project Architecture

## Overview

This project builds a **multimodal emotion recognition system** using **knowledge distillation** from Large Multimodal Models (LMMs). Lightweight student models are trained to replicate the reasoning of powerful teacher LMMs, enabling efficient inference from video clips.

**Emotion Classes (7):** Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise

**Modalities:** Text (transcripts), Audio (mel-spectrograms), Video (frames)

---

## High-Level Pipeline

```
Raw Videos
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION                                     │
│  download_videos.py → dataset_videos/ (2619 clips)      │
│  Scene detection (ContentDetector, threshold=27)        │
│  Filter: single face, English language                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  2. TEACHER ANNOTATION                                  │
│  annotate_videos.py                                     │
│  Teacher LMMs analyze each clip → emotion + rationale  │
│  Output: annotated_data.json + annotated_data.csv       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  3. DATASET GENERATION                                  │
│  generate_torch_dataset.py                              │
│  Split each clip into: text / audio / video modalities  │
│  Encode rationale text → 384D vectors (SBERT)           │
│  Output: checkpoints/datasets/multimodal_dataset_*.pt   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  4. STUDENT MODEL TRAINING                              │
│  Each student model learns to predict the 384D          │
│  rationale vector from its respective modality          │
│  Loss: MSE + Cosine Similarity                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  5. MULTIMODAL FUSION & CLASSIFICATION                  │
│  scripts/multimodal_emotion_model.py                    │
│  Fuse text/audio/video embeddings → 7-class prediction  │
└─────────────────────────────────────────────────────────┘
```

---

## Teacher Models (`models/teacher/`)

Three LMM teachers are supported, all accessed via API. Each receives a video clip and returns a structured JSON with emotion label, confidence score, and soft probability distribution over 7 classes.

| Model | Provider | File | API |
|---|---|---|---|
| Gemini 2.0 Flash | Google Vertex AI | `gemini.py` | Vertex AI SDK |
| Qwen 3.5-122B | NVIDIA | `qwen.py` | NVIDIA NIM API |
| Nemotron Nano 12B | NVIDIA | `nemotron.py` | NVIDIA NIM API |

**Active teacher** is controlled by `CURRENT_TEACHER_MODEL` in `.env`.

**Annotation flow:**
- `annotate_videos.py` runs up to 5 concurrent workers (`ThreadPoolExecutor`)
- Skips already-annotated files (cached in `annotated_data.json`)
- Filters by confidence threshold (0.85) before accepting pseudo-labels

---

## Rationale Encoder (`scripts/sentence_bert.py`)

```
Rationale Text (string)
        │
        ▼
  SentenceTransformer
  (all-MiniLM-L12-v2)
        │
        ▼
   384D Semantic Vector  ← r*_m (frozen target)
```

- Model: `all-MiniLM-L12-v2` (SBERT), **frozen** during all training
- Output dimension: **384**
- Serves as the stable supervision signal for all three student models
- Results are cached during dataset generation to avoid redundant encoding

---

## Student Models (`models/student/`)

All three student models share the same objective: **predict the 384D rationale vector** from their respective modality input. They all use the same combined loss function.

### Loss Function (shared across all students)

```
L_total = w_mse * MSE(predicted, target) + w_cosine * (1 - CosineSimilarity(predicted, target))
```

Default weights: `w_mse = 1.0`, `w_cosine = 0.5`  
*(Text model uses `w_mse = 0.0`, `w_cosine = 1.0` — cosine-only)*

---

### 1. Text Student — `BiLSTMTextToVector` (`text_bilstm.py`)

**Input:** Token IDs from BERT tokenizer (`bert-base-uncased`), max length 512  
**Output:** 384D vector

```
Token IDs (batch, 512)
        │
        ▼
  nn.Embedding(vocab_size, 128)   ← learned, not pretrained BERT weights
        │
   Dropout(0.3)
        │
        ▼
  BiLSTM(128→256, 2 layers, bidirectional)
        │  lstm_output: (batch, seq_len, 512)
        ▼
  Attention Pooling
  (Linear(512→1) → softmax → weighted sum)
        │  context: (batch, 512)
        ▼
  FC: Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→384)
        │
        ▼
   384D Output Vector
```

**Trainer:** `TextToVectorTrainer`  
**Optimizer:** Adam, lr=0.001  
**Note:** No gradient clipping in text trainer

---

### 2. Audio Student — `CNNBiLSTMAudioToVector` (`audio_cnn_bilstm.py`)

**Input:** Mel-spectrogram `(batch, 1, 128, T)` — 128 mel bands, variable time steps  
**Output:** 384D vector

```
Mel-Spectrogram (batch, 1, 128, T)
        │
        ▼
  Conv2D Block 1: Conv(1→32, 3×3) → BN → ReLU → MaxPool2d(2,2) → Dropout2d
        │  shape: (batch, 32, 64, T/2)
        ▼
  Conv2D Block 2: Conv(32→64, 3×3) → BN → ReLU → MaxPool2d(2,2) → Dropout2d
        │  shape: (batch, 64, 32, T/4)
        ▼
  Conv2D Block 3: Conv(64→128, 3×3) → BN → ReLU → MaxPool2d(2,2) → Dropout2d
        │  shape: (batch, 128, 16, T/8)
        ▼
  Reshape: permute → (batch, T/8, 128×16=2048)
        │
        ▼
  BiLSTM(2048→256, 2 layers, bidirectional)   [pack_padded_sequence for variable T]
        │  lstm_output: (batch, T/8, 512)
        ▼
  Attention Pooling (masked for padding)
        │  context: (batch, 512)
        ▼
  FC: Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→384)
        │
        ▼
   384D Output Vector
```

**Trainer:** `AudioToVectorTrainer`  
**Optimizer:** Adam, lr=0.001  
**Gradient clipping:** max_norm=1.0  
**Variable length handling:** `pack_padded_sequence` + length-adjusted masking

---

### 3. Video Student — `CNN3DBiLSTMVideoToVector` (`video_3dcnn_bilstm.py`)

**Input:** Video frames `(batch, 3, F, H, W)` — 3 channels, variable frames, spatial dims  
**Output:** 384D vector

```
Frames (batch, 3, F, H, W)
        │
        ▼
  Conv3D Block 1: Conv3d(3→64, 3×3×3) → BN3d → ReLU → MaxPool3d(1,2,2) → Dropout3d
        │  spatial halved, temporal preserved
        ▼
  Conv3D Block 2: Conv3d(64→128, 3×3×3) → BN3d → ReLU → MaxPool3d(1,2,2) → Dropout3d
        ▼
  Conv3D Block 3: Conv3d(128→256, 3×3×3) → BN3d → ReLU → MaxPool3d(1,2,2) → Dropout3d
        ▼
  Conv3D Block 4: Conv3d(256→512, 3×3×3) → BN3d → ReLU → MaxPool3d(1,2,2) → Dropout3d
        │  shape: (batch, 512, F, H/16, W/16)
        ▼
  Temporal Conv: Conv3d(512→512, 3×1×1) → BN3d → ReLU → MaxPool3d(2,1,1)
        │  temporal halved: (batch, 512, F/2, H/16, W/16)
        ▼
  AdaptiveAvgPool3d((None, 1, 1))   ← global spatial pooling
        │  shape: (batch, 512, F/2, 1, 1)
        ▼
  Reshape: squeeze → permute → (batch, F/2, 512)
        │
        ▼
  BiLSTM(512→512, 2 layers, bidirectional)   [pack_padded_sequence]
        │  lstm_output: (batch, F/2, 1024)
        ▼
  Attention Pooling (masked for padding)
        │  context: (batch, 1024)
        ▼
  FC: Linear(1024→512) → ReLU → Dropout → Linear(512→256) → ReLU → Dropout → Linear(256→384)
        │
        ▼
   384D Output Vector
```

**Trainer:** `VideoToVectorTrainer`  
**Optimizer:** Adam, lr=0.0001 (lower for 3D CNNs)  
**Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)  
**Gradient clipping:** max_norm=1.0

---

## Multimodal Fusion Model (`scripts/multimodal_emotion_model.py`)

A separate, lighter model used for final emotion classification. It uses simpler encoders (not the student models above) and fuses them with cross-modal attention.

```
Text Tokens ──► TinyTextEncoder (Embedding+GRU, 64D)  ──┐
Audio MFCCs ──► TinyAudioEncoder (1D-CNN, 64D)          ├──► CrossModalAttentionFusion
Video Landmarks ► TinyVideoEncoder (MLP on landmarks, 64D) ┘         │
                                                                       ▼
                                                          TransformerEncoderLayer (d=64, heads=4)
                                                                       │
                                                              mean pooling → 64D
                                                                       │
                                                          Classifier: Linear(64→32)→ReLU→Linear(32→7)
                                                                       │
                                                              7-class Emotion Logits
```

**Note:** This model uses MediaPipe face landmarks (468 points × 2 = 936D) for video, not raw frames.

---

## Semi-Supervised Training (`scripts/semi_supervised_pipeline.py`)

```
Stage 1: Pseudo-Label Generation
  Unlabeled data → Teacher LMM → confidence ≥ 0.85 → accepted pseudo-labels

Stage 2: Student Training with Knowledge Distillation
  Loss = α * CE(student_logits, hard_labels)
       + (1-α) * T² * KL_div(student_soft / T, teacher_soft / T)
  α = 0.7, T = 3.0 (temperature)
  Batch size: 1000, Epochs: 50
```

---

## Dataset Bundle (`checkpoints/datasets/`)

Saved as a `.pt` file with the following structure:

```python
{
    'text':   TextToVectorDataset,   # transcripts → 384D rationale vectors
    'audio':  AudioToVectorDataset,  # mel-spectrograms → 384D rationale vectors
    'video':  VideoToVectorDataset,  # frames → 384D rationale vectors
    'loader_params': {'batch_size': 2, 'shuffle': True}
}
```

---

## Video Processing (`common/video_helpers.py`)

Each raw video clip is processed into three modalities:

| Modality | Tool | Output |
|---|---|---|
| Audio | librosa | Mel-spectrogram (128 bands, variable T) |
| Text | Whisper (tiny) | Transcript string |
| Video | MoviePy + OpenCV | Frames at 1 fps, resized to 512×512 |

Scene detection uses `scenedetect` with `ContentDetector(threshold=27)`.  
Face detection uses MediaPipe (`min_detection_confidence=0.5`).

---

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini teacher |
| `GEMINI_PROJECT_ID` / `GEMINI_REGION` | Vertex AI project |
| `QWEN_API_KEY` / `QWEN_MODEL` | Qwen teacher |
| `NEMOTRON_API_KEY` / `NEMOTRON_MODEL` | Nemotron teacher |
| `CURRENT_TEACHER_MODEL` | Active teacher (`gemini`, `qwen`, `nemotron`) |
| `DOWNLOAD_LIMIT_PER_SOURCE` | Max videos per source |
| `BASE_DOWNLOAD_PATH` | Root path for raw video downloads |

---

## File Structure Summary

```
├── models/
│   ├── student/
│   │   ├── text_bilstm.py          # BiLSTM text → 384D
│   │   ├── audio_cnn_bilstm.py     # CNN-BiLSTM audio → 384D
│   │   └── video_3dcnn_bilstm.py   # 3D CNN-BiLSTM video → 384D
│   └── teacher/
│       ├── common.py               # Teacher router
│       ├── gemini.py               # Gemini Vertex AI
│       ├── qwen.py                 # Qwen NVIDIA NIM
│       └── nemotron.py             # Nemotron NVIDIA NIM
├── scripts/
│   ├── multimodal_emotion_model.py # Fusion model + tiny encoders
│   ├── semi_supervised_pipeline.py # KD training pipeline
│   └── sentence_bert.py            # Frozen SBERT rationale encoder
├── dataset_generation/
│   ├── download_videos.py          # yt-dlp downloader + scene splitter
│   ├── annotate_videos.py          # Parallel teacher annotation
│   └── generate_torch_dataset.py   # Build PyTorch dataset bundle
├── common/
│   ├── config.py                   # API keys + env vars
│   ├── constant.py                 # Emotion maps + model prompts
│   ├── video_helpers.py            # Modality extraction utilities
│   ├── csv_helper.py               # CSV read/write helpers
│   ├── file_helper.py              # Path + prompt file helpers
│   └── helper_functions.py         # Misc utilities
├── checkpoints/
│   ├── datasets/                   # Saved PyTorch dataset bundles
│   └── models/                     # Saved student model weights
├── assets/                         # Sample video clips
├── dataset_videos/                 # 2619 split video clips
├── Global_6000_Dataset/            # Raw downloaded videos by category
├── annotated_data.json             # Teacher annotation cache
├── annotated_data.csv              # Annotations as CSV
├── dataset_links.json              # Video source URLs by category
└── create_dataset.py               # Entry point: download → annotate → dataset
```
