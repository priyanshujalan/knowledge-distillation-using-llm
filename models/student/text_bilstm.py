import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TextToVectorDataset(Dataset):
    """Dataset for text-to-vector conversion"""

    def __init__(self, transcribed_texts, rationale_vectors, tokenizer, max_length=512):
        self.texts = transcribed_texts
        self.vectors = rationale_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        vector = self.vectors[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target_vector': torch.FloatTensor(vector)
        }


class BiLSTMTextToVector(nn.Module):
    """BiLSTM model to convert text to fixed-size vector"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=384, num_layers=2, dropout=0.3):
        super(BiLSTMTextToVector, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM layers
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism for pooling
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def attention_pooling(self, lstm_output, attention_mask):
        """Apply attention mechanism for sequence pooling"""
        # lstm_output: (batch, seq_len, hidden_dim*2)
        # attention_mask: (batch, seq_len)

        # Calculate attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_dim*2)
        ).squeeze(1)  # (batch, hidden_dim*2)

        return context_vector

    def forward(self, input_ids, attention_mask):
        # Embed input
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # BiLSTM
        lstm_output, (hidden, cell) = self.bilstm(embedded)
        # lstm_output: (batch, seq_len, hidden_dim*2)

        # Apply attention pooling
        pooled = self.attention_pooling(lstm_output, attention_mask)

        # Fully connected layers to output vector
        output_vector = self.fc(pooled)  # (batch, output_dim)

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

        # Cosine similarity loss (convert to loss by taking 1 - similarity)
        cosine_sim = self.cosine_similarity(predicted, target).mean()
        cosine_loss = 1 - cosine_sim

        # Combined loss
        total_loss = self.mse_weight * mse + self.cosine_weight * cosine_loss

        return total_loss, mse, cosine_sim


class VectorSimilarityEvaluator:
    """Evaluate vector similarity with random vectors"""

    def __init__(self, vector_dim=384):
        self.vector_dim = vector_dim

    def generate_random_vector(self, normalize=True):
        """Generate a random vector"""
        random_vec = np.random.randn(self.vector_dim)
        if normalize:
            random_vec = random_vec / np.linalg.norm(random_vec)
        return random_vec

    def compute_similarities(self, predicted_vector, target_vector, num_random=10):
        """
        Compute cosine similarities between:
        1. Predicted vs Target (should be high)
        2. Predicted vs Random vectors (should be low)
        3. Target vs Random vectors (baseline comparison)
        """
        # Ensure vectors are 2D for sklearn
        pred_vec = predicted_vector.reshape(1, -1)
        target_vec = target_vector.reshape(1, -1)

        # Similarity between predicted and target
        pred_target_sim = cosine_similarity(pred_vec, target_vec)[0, 0]

        # Generate random vectors and compute similarities
        random_vectors = np.array([self.generate_random_vector() for _ in range(num_random)])

        pred_random_sims = cosine_similarity(pred_vec, random_vectors)[0]
        target_random_sims = cosine_similarity(target_vec, random_vectors)[0]

        results = {
            'predicted_target_similarity': pred_target_sim,
            'predicted_random_similarities': {
                'mean': np.mean(pred_random_sims),
                'std': np.std(pred_random_sims),
                'min': np.min(pred_random_sims),
                'max': np.max(pred_random_sims),
                'all': pred_random_sims
            },
            'target_random_similarities': {
                'mean': np.mean(target_random_sims),
                'std': np.std(target_random_sims),
                'min': np.min(target_random_sims),
                'max': np.max(target_random_sims),
                'all': target_random_sims
            }
        }

        return results

    def print_similarity_report(self, results):
        """Print a formatted similarity report"""
        print("\n" + "=" * 60)
        print("VECTOR SIMILARITY ANALYSIS")
        print("=" * 60)
        print(f"\nPredicted vs Target Similarity: {results['predicted_target_similarity']:.4f}")
        print("\nPredicted vs Random Vectors:")
        print(f"  Mean: {results['predicted_random_similarities']['mean']:.4f}")
        print(f"  Std:  {results['predicted_random_similarities']['std']:.4f}")
        print(f"  Min:  {results['predicted_random_similarities']['min']:.4f}")
        print(f"  Max:  {results['predicted_random_similarities']['max']:.4f}")
        print("\nTarget vs Random Vectors (Baseline):")
        print(f"  Mean: {results['target_random_similarities']['mean']:.4f}")
        print(f"  Std:  {results['target_random_similarities']['std']:.4f}")
        print(f"  Min:  {results['target_random_similarities']['min']:.4f}")
        print(f"  Max:  {results['target_random_similarities']['max']:.4f}")

        # Quality assessment
        pred_target = results['predicted_target_similarity']
        pred_random_mean = results['predicted_random_similarities']['mean']

        print("\n" + "-" * 60)
        print("QUALITY ASSESSMENT:")
        if pred_target > 0.8:
            print("  ✓ EXCELLENT: Predicted vector is very similar to target")
        elif pred_target > 0.6:
            print("  ✓ GOOD: Predicted vector is reasonably similar to target")
        elif pred_target > 0.4:
            print("  ⚠ MODERATE: Predicted vector has moderate similarity to target")
        else:
            print("  ✗ POOR: Predicted vector is not similar to target")

        if pred_target > pred_random_mean + 0.2:
            print("  ✓ Predicted vector is significantly better than random")
        else:
            print("  ✗ Predicted vector is not much better than random")
        print("=" * 60 + "\n")


class TextToVectorTrainer:
    """Training wrapper for the BiLSTM model"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.evaluator = VectorSimilarityEvaluator()

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_cosine_sim = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_vectors = batch['target_vector'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            predicted_vectors = self.model(input_ids, attention_mask)

            # Calculate loss
            loss, mse, cosine_sim = criterion(predicted_vectors, target_vectors)

            # Backward pass
            loss.backward()
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_vectors = batch['target_vector'].to(self.device)

                predicted_vectors = self.model(input_ids, attention_mask)
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

    def predict(self, text, tokenizer, max_length=512):
        """Predict vector for a single text"""
        self.model.eval()

        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            predicted_vector = self.model(input_ids, attention_mask)

        return predicted_vector.cpu().numpy()[0]

    def evaluate_prediction(self, text, target_vector, tokenizer, num_random=10):
        """Predict and evaluate against random vectors"""
        predicted_vector = self.predict(text, tokenizer)

        results = self.evaluator.compute_similarities(
            predicted_vector,
            target_vector,
            num_random=num_random
        )

        self.evaluator.print_similarity_report(results)

        return predicted_vector, results


def load_data_and_train_for_text(model, train_loader, num_epochs=10):
    trainer = TextToVectorTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CombinedLoss(mse_weight=0, cosine_weight=1)

    # Training loop
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, optimizer, criterion)
        # val_metrics = trainer.validate(val_loader, criterion)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Cosine Sim: {train_metrics['cosine_similarity']:.4f}")
            # print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
            #       f"MSE: {val_metrics['mse']:.4f}, "
            #       f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}")

    return trainer

# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Sample data (replace with your actual data)
    transcribed_texts = [
        "This is a sample transcribed audio text",
        "Another example of transcribed speech",
        "Machine learning helps us understand data patterns",
        "Deep learning models are powerful tools"
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
    train_texts = transcribed_texts[:3]
    train_vectors = rationale_vectors[:3]
    val_texts = transcribed_texts[3:]
    val_vectors = rationale_vectors[3:]

    # Create datasets
    train_dataset = TextToVectorDataset(train_texts, train_vectors, tokenizer)
    val_dataset = TextToVectorDataset(val_texts, val_vectors, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Initialize model
    model = BiLSTMTextToVector(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=384,
        num_layers=2,
        dropout=0.3
    )

    # Training setup
    trainer = TextToVectorTrainer(model)
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

    # Test prediction with similarity evaluation
    test_text = "New transcribed audio text for testing"
    test_target = np.random.randn(384)
    test_target = test_target / np.linalg.norm(test_target)

    print("\n" + "=" * 60)
    print("TESTING PREDICTION")
    print("=" * 60)
    predicted_vector, similarity_results = trainer.evaluate_prediction(
        test_text,
        test_target,
        tokenizer,
        num_random=20
    )

    print(f"\nPredicted vector shape: {predicted_vector.shape}")
    print(f"Predicted vector norm: {np.linalg.norm(predicted_vector):.4f}")