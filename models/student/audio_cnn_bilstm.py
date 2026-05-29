import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class AudioToVectorDataset(Dataset):
    """Dataset for mel spectrogram to vector conversion"""

    def __init__(self, mel_spectrograms, rationale_vectors):
        """
        Args:
            mel_spectrograms: List of mel spectrograms, each of shape (128, time_steps)
            rationale_vectors: List of target vectors, each of shape (384,)
        """
        self.mel_spectrograms = mel_spectrograms
        self.vectors = rationale_vectors

    def __len__(self):
        return len(self.mel_spectrograms)

    def __getitem__(self, idx):
        mel_spec = self.mel_spectrograms[idx]
        vector = self.vectors[idx]

        # Ensure mel_spec is (1, 128, time_steps) - add channel dimension
        if len(mel_spec.shape) == 2:
            mel_spec = np.expand_dims(mel_spec, axis=0)

        return {
            'mel_spectrogram': torch.FloatTensor(mel_spec),
            'target_vector': torch.FloatTensor(vector)
        }


def collate_fn(batch):
    """Custom collate function to handle variable length spectrograms"""
    # Find max time steps in batch
    max_time = max([item['mel_spectrogram'].shape[2] for item in batch])

    # Pad all spectrograms to max_time
    mel_specs = []
    lengths = []

    for item in batch:
        mel_spec = item['mel_spectrogram']
        time_steps = mel_spec.shape[2]
        lengths.append(time_steps)

        # Pad if necessary
        if time_steps < max_time:
            padding = torch.zeros(1, 128, max_time - time_steps)
            mel_spec = torch.cat([mel_spec, padding], dim=2)

        mel_specs.append(mel_spec)

    # Stack into batch
    mel_specs = torch.stack(mel_specs)
    target_vectors = torch.stack([item['target_vector'] for item in batch])
    lengths = torch.LongTensor(lengths)

    return {
        'mel_spectrogram': mel_specs,
        'target_vector': target_vectors,
        'lengths': lengths
    }


class CNNBiLSTMAudioToVector(nn.Module):
    """CNN + BiLSTM model to convert mel spectrogram to fixed-size vector"""

    def __init__(self, output_dim=384, cnn_channels=[32, 64, 128],
                 lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super(CNNBiLSTMAudioToVector, self).__init__()

        # CNN layers for feature extraction from mel spectrogram
        # Input: (batch, 1, 128, time_steps)
        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),  # Reduce both mel and time dimensions
                    nn.Dropout2d(dropout)
                )
            )
            in_channels = out_channels

        # Calculate the feature dimension after CNN layers
        # After 3 pooling layers: 128 / 2^3 = 16 mel bands
        self.mel_reduced = 128 // (2 ** len(cnn_channels))
        self.cnn_output_dim = cnn_channels[-1] * self.mel_reduced

        # BiLSTM layers
        self.bilstm = nn.LSTM(
            self.cnn_output_dim,
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
            nn.Linear(lstm_hidden, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, mel_spectrogram, lengths=None):
        """
        Args:
            mel_spectrogram: (batch, 1, 128, time_steps)
            lengths: (batch,) actual lengths before padding
        Returns:
            output_vector: (batch, output_dim)
        """
        batch_size = mel_spectrogram.size(0)

        # CNN feature extraction
        x = mel_spectrogram
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # x shape: (batch, cnn_channels[-1], mel_reduced, time_reduced)

        # Reshape for LSTM: (batch, time, features)
        # Flatten mel dimension into feature dimension
        batch, channels, mel, time = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, mel)
        x = x.contiguous().view(batch, time, channels * mel)  # (batch, time, features)

        # BiLSTM
        if lengths is not None:
            # Adjust lengths based on pooling
            adjusted_lengths = lengths // (2 ** len(self.conv_layers))
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
            adjusted_lengths = lengths // (2 ** len(self.conv_layers))
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


class AudioToVectorTrainer:
    """Training wrapper for the CNN-BiLSTM model"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_cosine_sim = 0

        for batch in dataloader:
            mel_specs = batch['mel_spectrogram'].to(self.device)
            target_vectors = batch['target_vector'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            predicted_vectors = self.model(mel_specs, lengths)

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
                mel_specs = batch['mel_spectrogram'].to(self.device)
                target_vectors = batch['target_vector'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                predicted_vectors = self.model(mel_specs, lengths)
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

    def predict(self, mel_spectrogram):
        """Predict vector for a single mel spectrogram"""
        self.model.eval()

        # Ensure proper shape (1, 1, 128, time_steps)
        if len(mel_spectrogram.shape) == 2:
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        if len(mel_spectrogram.shape) == 3:
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

        mel_tensor = torch.FloatTensor(mel_spectrogram).to(self.device)

        with torch.no_grad():
            predicted_vector = self.model(mel_tensor)

        return predicted_vector.cpu().numpy()[0]


def load_data_and_train_for_audio(model, dataloader, num_epochs):
    trainer = AudioToVectorTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CombinedLoss(mse_weight=1.0, cosine_weight=0.5)
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(dataloader, optimizer, criterion)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Cosine Sim: {train_metrics['cosine_similarity']:.4f}")

    return trainer

# Example usage
if __name__ == "__main__":
    # Sample data - replace with your actual mel spectrograms
    # Each mel spectrogram has shape (128, time_steps)
    mel_spectrograms = [
        np.random.randn(128, 200),  # Different time lengths
        np.random.randn(128, 250),
        np.random.randn(128, 180),
        np.random.randn(128, 220)
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
    train_mels = mel_spectrograms[:3]
    train_vectors = rationale_vectors[:3]
    val_mels = mel_spectrograms[3:]
    val_vectors = rationale_vectors[3:]

    # Create datasets
    train_dataset = AudioToVectorDataset(train_mels, train_vectors)
    val_dataset = AudioToVectorDataset(val_mels, val_vectors)

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    model = CNNBiLSTMAudioToVector(
        output_dim=384,
        cnn_channels=[32, 64, 128],
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3
    )

    # Print model architecture
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    trainer = AudioToVectorTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CombinedLoss(mse_weight=1.0, cosine_weight=0.5)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, optimizer, criterion)
        val_metrics = trainer.validate(val_loader, criterion)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Cosine Sim: {train_metrics['cosine_similarity']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"MSE: {val_metrics['mse']:.4f}, "
                  f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}")

    # Prediction
    test_mel = np.random.randn(128, 190)
    predicted_vector = trainer.predict(test_mel)
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