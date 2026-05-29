import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VideoToVectorDataset(Dataset):
    """Dataset for video frames to vector conversion"""

    def __init__(self, video_frames, rationale_vectors, transform=None):
        """
        Args:
            video_frames: List of video frame sequences, each of shape (num_frames, H, W, C) or (num_frames, C, H, W)
            rationale_vectors: List of target vectors, each of shape (384,)
            transform: Optional transform to apply to frames
        """
        self.video_frames = video_frames
        self.vectors = rationale_vectors
        self.transform = transform

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frames = self.video_frames[idx]
        vector = self.vectors[idx]

        # Ensure frames are in (C, num_frames, H, W) format for 3D CNN
        # If input is (num_frames, H, W, C), convert to (C, num_frames, H, W)
        if frames.shape[-1] == 3 or frames.shape[-1] == 1:  # Channel last
            frames = np.transpose(frames, (3, 0, 1, 2))
        # If input is (num_frames, C, H, W), convert to (C, num_frames, H, W)
        elif len(frames.shape) == 4 and frames.shape[1] in [1, 3]:
            frames = np.transpose(frames, (1, 0, 2, 3))

        # Normalize frames to [0, 1] if not already
        if frames.max() > 1.0:
            frames = frames / 255.0

        if self.transform:
            frames = self.transform(frames)

        return {
            'frames': torch.FloatTensor(frames),
            'target_vector': torch.FloatTensor(vector)
        }


def collate_fn_for_video(batch):
    """Custom collate function to handle variable number of frames"""
    # Find max number of frames in batch
    max_frames = max([item['frames'].shape[1] for item in batch])

    # Get dimensions from first item
    C, _, H, W = batch[0]['frames'].shape

    # Pad all videos to max_frames
    padded_frames = []
    lengths = []

    for item in batch:
        frames = item['frames']
        num_frames = frames.shape[1]
        lengths.append(num_frames)

        # Pad if necessary
        if num_frames < max_frames:
            padding = torch.zeros(C, max_frames - num_frames, H, W)
            frames = torch.cat([frames, padding], dim=1)

        padded_frames.append(frames)

    # Stack into batch
    padded_frames = torch.stack(padded_frames)
    target_vectors = torch.stack([item['target_vector'] for item in batch])
    lengths = torch.LongTensor(lengths)

    return {
        'frames': padded_frames,
        'target_vector': target_vectors,
        'lengths': lengths
    }


class CNN3DBiLSTMVideoToVector(nn.Module):
    """3D CNN + BiLSTM model to convert video frames to fixed-size vector"""

    def __init__(self, input_channels=3, output_dim=384,
                 cnn_channels=[64, 128, 256, 512],
                 lstm_hidden=512, lstm_layers=2, dropout=0.3):
        super(CNN3DBiLSTMVideoToVector, self).__init__()

        # 3D CNN layers for spatio-temporal feature extraction
        # Input: (batch, C, num_frames, H, W)
        self.conv3d_layers = nn.ModuleList()
        in_channels = input_channels

        for out_channels in cnn_channels:
            self.conv3d_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels, out_channels,
                        kernel_size=(3, 3, 3),
                        padding=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=(1, 2, 2)),  # Pool spatial, keep temporal
                    nn.Dropout3d(dropout)
                )
            )
            in_channels = out_channels

        # Additional temporal conv to reduce frame dimension
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                cnn_channels[-1], cnn_channels[-1],
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(cnn_channels[-1]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1))  # Pool temporal dimension
        )

        # Global average pooling for spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep temporal, pool spatial

        # BiLSTM layers for temporal modeling
        self.bilstm = nn.LSTM(
            cnn_channels[-1],
            lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Attention mechanism for temporal pooling
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, frames, lengths=None):
        """
        Args:
            frames: (batch, C, num_frames, H, W)
            lengths: (batch,) actual number of frames before padding
        Returns:
            output_vector: (batch, output_dim)
        """
        batch_size = frames.size(0)

        # 3D CNN feature extraction
        x = frames
        for conv_layer in self.conv3d_layers:
            x = conv_layer(x)
        # x shape: (batch, cnn_channels[-1], num_frames, H_reduced, W_reduced)

        # Additional temporal convolution
        x = self.temporal_conv(x)
        # x shape: (batch, cnn_channels[-1], num_frames_reduced, H_reduced, W_reduced)

        # Global average pooling over spatial dimensions
        x = self.global_avg_pool(x)
        # x shape: (batch, cnn_channels[-1], num_frames_reduced, 1, 1)

        # Reshape for LSTM: (batch, time, features)
        x = x.squeeze(-1).squeeze(-1)  # (batch, cnn_channels[-1], num_frames_reduced)
        x = x.permute(0, 2, 1)  # (batch, num_frames_reduced, cnn_channels[-1])

        # BiLSTM
        if lengths is not None:
            # Adjust lengths based on temporal pooling (pooled twice: temporal_conv has pool of 2)
            adjusted_lengths = lengths // 2
            adjusted_lengths = torch.clamp(adjusted_lengths, min=1)

            # Pack padded sequence
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, adjusted_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.bilstm(packed_input)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.bilstm(x)

        # lstm_output: (batch, time, lstm_hidden*2)

        # Attention pooling
        attention_scores = self.attention(lstm_output).squeeze(-1)  # (batch, time)

        # Create mask for padded positions if lengths provided
        if lengths is not None:
            adjusted_lengths = lengths // 2
            adjusted_lengths = torch.clamp(adjusted_lengths, min=1)
            max_len = lstm_output.size(1)
            mask = torch.arange(max_len, device=lstm_output.device).expand(
                batch_size, max_len
            ) < adjusted_lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, time)

        # Weighted sum
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, time)
            lstm_output  # (batch, time, lstm_hidden*2)
        ).squeeze(1)  # (batch, lstm_hidden*2)

        # Fully connected to output vector
        output_vector = self.fc(context_vector)  # (batch, output_dim)

        return output_vector


class CombinedLoss(nn.Module):
    """Combined MSE and Cosine Similarity Loss"""

    def __init__(self, mse_weight=1.0, cosine_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, predicted, target):
        # MSE loss
        mse = self.mse_loss(predicted, target)

        # Cosine similarity loss
        cosine_sim = self.cosine_similarity(predicted, target).mean()
        cosine_loss = 1 - cosine_sim

        # Combined loss
        total_loss = self.mse_weight * mse + self.cosine_weight * cosine_loss

        return total_loss, mse, cosine_sim


class VideoToVectorTrainer:
    """Training wrapper for the 3D CNN-BiLSTM model"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_cosine_sim = 0

        for batch in dataloader:
            frames = batch['frames'].to(self.device)
            target_vectors = batch['target_vector'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            predicted_vectors = self.model(frames, lengths)

            # Calculate loss
            loss, mse, cosine_sim = criterion(predicted_vectors, target_vectors)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_cosine_sim += cosine_sim.item()

        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'mse': total_mse / n_batches,
            'cosine_similarity': total_cosine_sim / n_batches
        }

    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_cosine_sim = 0

        with torch.no_grad():
            for batch in dataloader:
                frames = batch['frames'].to(self.device)
                target_vectors = batch['target_vector'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                predicted_vectors = self.model(frames, lengths)
                loss, mse, cosine_sim = criterion(predicted_vectors, target_vectors)

                total_loss += loss.item()
                total_mse += mse.item()
                total_cosine_sim += cosine_sim.item()

        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'mse': total_mse / n_batches,
            'cosine_similarity': total_cosine_sim / n_batches
        }

    def predict(self, frames):
        """Predict vector for a single video"""
        self.model.eval()

        # Ensure proper shape (1, C, num_frames, H, W)
        if len(frames.shape) == 4:
            # If (num_frames, H, W, C), convert
            if frames.shape[-1] == 3 or frames.shape[-1] == 1:
                frames = np.transpose(frames, (3, 0, 1, 2))
            # If (num_frames, C, H, W), convert
            elif frames.shape[1] in [1, 3]:
                frames = np.transpose(frames, (1, 0, 2, 3))

        # Add batch dimension
        if len(frames.shape) == 4:
            frames = np.expand_dims(frames, axis=0)

        # Normalize if needed
        if frames.max() > 1.0:
            frames = frames / 255.0

        frames_tensor = torch.FloatTensor(frames).to(self.device)

        with torch.no_grad():
            predicted_vector = self.model(frames_tensor)

        return predicted_vector.cpu().numpy()[0]


def load_data_and_train_for_video(model, dataloader, num_epochs=20):

    trainer = VideoToVectorTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for 3D CNNs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = CombinedLoss(mse_weight=1.0, cosine_weight=0.5)

    # Training loop
    # best_val_loss = float('inf')

    print("Video Training Starts here...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} Time: {time.time()}")
        train_metrics = trainer.train_epoch(dataloader, optimizer, criterion)
        # val_metrics = trainer.validate(val_loader, criterion)

        # Learning rate scheduling
        scheduler.step(train_metrics['loss']) # val_metrics['loss']

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Cosine Sim: {train_metrics['cosine_similarity']:.4f}")
            # print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
            #       f"MSE: {val_metrics['mse']:.4f}, "
            #       f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}")

        # Save best model
        # if val_metrics['loss'] < best_val_loss:
        #     best_val_loss = val_metrics['loss']
        #     torch.save(model.state_dict(), 'checkpoints/best_video_model.pth')
        #
    return trainer


# Example usage
if __name__ == "__main__":
    # Sample data - replace with your actual video frames
    # Each video has shape (num_frames, H, W, C) or (num_frames, C, H, W)
    # For this example: (num_frames, 224, 224, 3)
    video_frames = [
        np.random.randint(0, 255, (30, 224, 224, 3), dtype=np.uint8),  # 30 frames
        np.random.randint(0, 255, (45, 224, 224, 3), dtype=np.uint8),  # 45 frames
        np.random.randint(0, 255, (25, 224, 224, 3), dtype=np.uint8),  # 25 frames
        np.random.randint(0, 255, (40, 224, 224, 3), dtype=np.uint8),  # 40 frames
    ]

    rationale_vectors = [
        np.random.randn(384),
        np.random.randn(384),
        np.random.randn(384),
        np.random.randn(384)
    ]

    # Normalize vectors (optional but recommended)
    rationale_vectors = [v / np.linalg.norm(v) for v in rationale_vectors]

    # Split into train/val
    train_videos = video_frames[:3]
    train_vectors = rationale_vectors[:3]
    val_videos = video_frames[3:]
    val_vectors = rationale_vectors[3:]

    # Create datasets
    train_dataset = VideoToVectorDataset(train_videos, train_vectors)
    val_dataset = VideoToVectorDataset(val_videos, val_vectors)

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_for_video
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_for_video
    )

    # Initialize model
    model = CNN3DBiLSTMVideoToVector(
        input_channels=3,
        output_dim=384,
        cnn_channels=[64, 128, 256, 512],
        lstm_hidden=512,
        lstm_layers=2,
        dropout=0.3
    )

    # Print model architecture
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    trainer = VideoToVectorTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for 3D CNNs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = CombinedLoss(mse_weight=1.0, cosine_weight=0.5)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, optimizer, criterion)
        val_metrics = trainer.validate(val_loader, criterion)

        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Cosine Sim: {train_metrics['cosine_similarity']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"MSE: {val_metrics['mse']:.4f}, "
                  f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'best_video_model.pth')

    # Prediction
    test_video = np.random.randint(0, 255, (35, 224, 224, 3), dtype=np.uint8)
    predicted_vector = trainer.predict(test_video)
    print(f"\nPredicted vector shape: {predicted_vector.shape}")

    # Generate random vector and compute cosine similarity
    random_vector = np.random.randn(384)
    random_vector = random_vector / np.linalg.norm(random_vector)

    # Compute cosine similarity
    cos_sim = cosine_similarity(
        predicted_vector.reshape(1, -1),
        random_vector.reshape(1, -1)
    )[0, 0]

    print(f"Cosine similarity with random vector: {cos_sim:.4f}")