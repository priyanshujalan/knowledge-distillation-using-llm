import torch
import torch.nn as nn


class TinyTextEncoder(nn.Module):
    """
    Lightweight Text Encoder using Embedding + GRU.
    Input: Integer token IDs (Batch, Seq_Len)
    Output: Sentence vector (Batch, embed_dim)
    """

    def __init__(self, vocab_size=30522, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # GRU is lighter than LSTM and often performs equally well for short text
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (Batch, Seq_Len)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # Run GRU
        _, hidden = self.gru(embedded)

        # hidden: (1, Batch, Hidden) -> squeeze to (Batch, Hidden)
        return hidden.squeeze(0)


class TinyAudioEncoder(nn.Module):
    """
    Lightweight 1D-CNN for Audio (e.g., MFCCs).
    Input: Audio features (Batch, Channels, TimeSteps)
    Output: Audio vector (Batch, output_dim)
    """

    def __init__(self, input_channels=40, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling to get fixed size vector
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        features = self.net(x)
        features = features.squeeze(-1)  # Remove time dim -> (Batch, 64)
        return self.fc(features)


class TinyVideoEncoder(nn.Module):
    """
    MLP for Pre-extracted Video Features (e.g., MediaPipe Face Landmarks).
    We assume input is a flattened vector of landmarks (e.g., 468 points * 2 coords = 936).
    Input: Landmark vector (Batch, input_dim)
    Output: Video vector (Batch, output_dim)
    """

    def __init__(self, input_dim=936, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class CrossModalAttentionFusion(nn.Module):
    """
    The Fusion 'Brain'.
    Uses Self-Attention to allow Text, Audio, and Video to 'talk' to each other.
    """

    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()

        # Project all inputs to the same dimension if they aren't already
        # (In this example they are all 64, but good practice to have projections)
        self.proj_text = nn.Linear(input_dim, hidden_dim)
        self.proj_audio = nn.Linear(input_dim, hidden_dim)
        self.proj_video = nn.Linear(input_dim, hidden_dim)

        # A single Transformer Encoder Layer for fusion
        # batch_first=True means input is (Batch, Seq, Feature)
        self.attention_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)

    def forward(self, text_emb, audio_emb, video_emb):
        # 1. Project inputs
        t = self.proj_text(text_emb).unsqueeze(1)  # (Batch, 1, Hidden)
        a = self.proj_audio(audio_emb).unsqueeze(1)  # (Batch, 1, Hidden)
        v = self.proj_video(video_emb).unsqueeze(1)  # (Batch, 1, Hidden)

        # 2. Stack as a sequence: [Text, Audio, Video]
        # Shape: (Batch, 3, Hidden)
        fusion_input = torch.cat([t, a, v], dim=1)

        # 3. Apply Self-Attention
        # The model figures out: "When Text is X, pay attention to Audio Y"
        attended_features = self.attention_layer(fusion_input)

        # 4. Pooling
        # We can average the 3 modality vectors to get a single 'Fused' vector
        fused_vector = attended_features.mean(dim=1)  # (Batch, Hidden)

        return fused_vector


class MultimodalEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Encoders
        self.text_encoder = TinyTextEncoder(vocab_size=10000, embed_dim=64, hidden_dim=64)
        self.audio_encoder = TinyAudioEncoder(input_channels=40, output_dim=64)  # e.g., 40 MFCCs
        self.video_encoder = TinyVideoEncoder(input_dim=936, output_dim=64)  # e.g., 468 * 2 (x,y)

        # Fusion
        self.fusion = CrossModalAttentionFusion(input_dim=64, hidden_dim=64)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # Output Logits
        )

    def forward(self, text_input, audio_input, video_input, apply_noise=False):
        """
        Forward pass.
        Args:
            apply_noise (bool): For Semi-Supervised Training (Stage 2).
                                If True, adds noise to features to force robustness.
        """

        # 1. Encode Unimodal Features
        t_emb = self.text_encoder(text_input)
        a_emb = self.audio_encoder(audio_input)
        v_emb = self.video_encoder(video_input)

        # --- Optional: Feature Noise Injection for Semi-Supervised Learning ---
        if apply_noise and self.training:
            # Example: Add random Gaussian noise to audio/video embeddings
            noise_a = torch.randn_like(a_emb) * 0.1
            noise_v = torch.randn_like(v_emb) * 0.1
            a_emb = a_emb + noise_a
            v_emb = v_emb + noise_v
            # Text could assume Dropout handles it, or mask specific dimensions
        # ----------------------------------------------------------------------

        # 2. Fuse
        fused_emb = self.fusion(t_emb, a_emb, v_emb)

        # 3. Classify
        logits = self.classifier(fused_emb)

        return logits
