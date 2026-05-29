import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super(ResidualBlock, self).__init__()

        # "Same" padding for dilated convolutions to preserve length
        # formula: padding = (kernel_size - 1) * dilation / 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.relu(out + residual)


class TCNRoBertaAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_blocks=4, kernel_size=3):
        super(TCNRoBertaAdapter, self).__init__()

        # Input Projection (768 -> 256)
        self.entry_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Stacked Residual Blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 1, 2, 4, 8...
            self.blocks.append(ResidualBlock(hidden_dim, kernel_size, dilation))

        # Output Projection (256 -> 768)
        self.exit_conv = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim) -> Need (Batch, Dim, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.entry_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.exit_conv(x)

        # Permute back: (Batch, Dim, Seq_Len) -> (Batch, Seq_Len, Dim)
        return x.permute(0, 2, 1)

class EmbeddingDataset(Dataset):
    def __init__(self, num_samples, seq_len, dim):
        # Replace this with loading your actual .pt files
        self.inputs = torch.randn(num_samples, seq_len, dim)
        self.targets = torch.randn(num_samples, seq_len, dim)  # Target embeddings

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def calculate_cosine_similarity(pred, target):
    """
    Computes average cosine similarity between two batches of sequences.
    Shape: (Batch, Seq, Dim)
    Returns: scalar value (1.0 = perfect alignment, 0.0 = orthogonal)
    """
    # Flatten to (Batch * Seq, Dim) to compute pairwise similarity easily
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    return F.cosine_similarity(pred_flat, target_flat).mean().item()

def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print(f"\n[Info] Starting TCN Training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0

        # Training Loop
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation Loop (Inline for brevity)
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss_accum += criterion(outputs, targets).item()

        avg_val_loss = val_loss_accum / len(val_loader)

        print(f"Epoch {epoch + 1}: Train MSE: {avg_train_loss:.5f} | Val MSE: {avg_val_loss:.5f}")

def test_model(model, test_loader, device):
    """
    Separate testing module that calculates MSE and Cosine Similarity.
    """
    model.eval()
    test_loss_accum = 0.0
    cosine_sim_accum = 0.0
    criterion = nn.MSELoss()

    print("\n[Info] Starting Testing...")

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Calculate MSE
            loss = criterion(outputs, targets)
            test_loss_accum += loss.item()

            # Calculate Cosine Similarity
            cosine_sim_accum += calculate_cosine_similarity(outputs, targets)

    avg_test_loss = test_loss_accum / len(test_loader)
    avg_cosine = cosine_sim_accum / len(test_loader)

    print("-" * 30)
    print("      TEST RESULTS      ")
    print("-" * 30)
    print(f"MSE Loss (Lower is better):       {avg_test_loss:.5f}")
    print(f"Cosine Sim (Higher is better):    {avg_cosine:.5f}")
    print("-" * 30)


if __name__ == "__main__":
    # Settings
    BATCH_SIZE = 32
    SEQ_LEN = 64
    DIM = 768
    EPOCHS = 3
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Data
    # In practice, split your data: 80% Train, 10% Val, 10% Test
    full_dataset = EmbeddingDataset(num_samples=1000, seq_len=SEQ_LEN, dim=DIM)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # We use the same set for val/test just for this demo
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model
    model = TCNRoBertaAdapter(input_dim=DIM, hidden_dim=256, num_blocks=4).to(DEVICE)

    # 3. Train
    train_model(model, train_loader, test_loader, EPOCHS, LR, DEVICE)

    # 4. Test
    test_model(model, test_loader, DEVICE)