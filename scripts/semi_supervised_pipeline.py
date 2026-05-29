import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os

from common.constant import EMOTIONS_MAP, EMOTIONS
from scripts.multimodal_emotion_model import MultimodalEmotionModel


# ==========================================
# 1. LLM Interface (Teacher)
# ==========================================

def query_llm_teacher(text_content, audio_path, video_path):
    """
    Simulates querying a SOTA LLM (e.g., Gemini 1.5 Pro, GPT-4o).
    """

    # 1. Pick a winner
    winner_idx = random.randint(0, len(EMOTIONS) - 1)
    predicted_emotion = EMOTIONS[winner_idx]

    # 2. Generate high confidence for the winner (0.80 to 0.99)
    high_confidence = random.uniform(0.80, 0.99)

    # 3. Distribute the remaining probability among others
    remaining_prob = 1.0 - high_confidence
    others_count = len(EMOTIONS) - 1

    soft_labels = [0.0] * len(EMOTIONS)
    soft_labels[winner_idx] = high_confidence

    current_sum = 0
    for i in range(len(EMOTIONS)):
        if i != winner_idx:
            share = remaining_prob / others_count
            noise = random.uniform(-share / 2, share / 2)
            val = max(0.0, share + noise)
            soft_labels[i] = val
            current_sum += val

    total_prob = sum(soft_labels)
    soft_labels = [x / total_prob for x in soft_labels]
    max_conf = max(soft_labels)

    return {
        "emotion": predicted_emotion,
        "confidence": max_conf,
        "soft_labels": soft_labels,
        "class_order": EMOTIONS
    }


# ==========================================
# 2. Data Handling
# ==========================================

class PseudoLabeledDataset(Dataset):
    def __init__(self, data_samples, emotion_map):
        self.data = data_samples
        self.emotion_map = emotion_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        text_tensor = sample['text_features']
        audio_tensor = sample['audio_features']
        video_tensor = sample['video_features']
        label_id = self.emotion_map[sample['label']]
        soft_labels = torch.tensor(sample['soft_labels'], dtype=torch.float)

        return text_tensor, audio_tensor, video_tensor, torch.tensor(label_id, dtype=torch.long), soft_labels


def generate_pseudo_labels(unlabeled_data, confidence_threshold=0.85):
    print(f"\n--- 🤖 Stage 1: Generating Pseudo-Labels (Threshold: {confidence_threshold}) ---")

    labeled_samples = []

    for i, sample in enumerate(unlabeled_data):
        txt = "dummy text transcript"
        aud = "path/to/audio.wav"
        vid = "path/to/video.mp4"

        response = query_llm_teacher(txt, aud, vid)

        conf = response['confidence']
        emotion = response['emotion']
        soft_labels = response['soft_labels']

        if conf >= confidence_threshold:
            print(f"  [Sample {i}] Accepted: {emotion} (Conf: {conf:.2f})")

            labeled_samples.append({
                'text_features': sample['text_feat'],
                'audio_features': sample['audio_feat'],
                'video_features': sample['video_feat'],
                'label': emotion,
                'soft_labels': soft_labels
            })
        else:
            print(f"  [Sample {i}] Rejected (Conf: {conf:.2f} < {confidence_threshold})")

    print(f"✅ Generated {len(labeled_samples)} high-quality pseudo-labels out of {len(unlabeled_data)} samples.")
    return labeled_samples


# ==========================================
# 3. Training Loop with Unified Loss
# ==========================================

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, hard_labels, teacher_probs):
        loss_ce = self.ce_loss(student_logits, hard_labels)
        student_log_soft = F.log_softmax(student_logits / self.T, dim=1)
        loss_kl = self.kl_div_loss(student_log_soft, teacher_probs)
        total_loss = (self.alpha * loss_ce) + ((1 - self.alpha) * (self.T ** 2) * loss_kl)
        return total_loss


def train_student_model(labeled_data, num_epochs=5, batch_size=4, save_path="student_emotion_model.pth"):
    print("\n--- 🎓 Stage 2: Training Student Model (Unified KD Loss) ---")

    dataset = PseudoLabeledDataset(labeled_data, EMOTIONS_MAP)

    if len(dataset) < 2:
        print("❌ Not enough data for Batch Normalization (Need > 1 sample). Training aborted.")
        return None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if len(dataloader) == 0:
        print("⚠️ Warning: Dataset size is smaller than batch size with drop_last=True.")
        print("   -> Switching to drop_last=False but ensure len(dataset) > 1")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    student_model = MultimodalEmotionModel(num_classes=7)
    student_model.train()

    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    kd_criterion = DistillationLoss(temperature=3.0, alpha=0.7)

    for epoch in range(num_epochs):
        total_loss = 0
        batches_processed = 0

        for batch in dataloader:
            t_in, a_in, v_in, hard_labels, soft_labels = batch

            if t_in.size(0) < 2:
                continue

            optimizer.zero_grad()
            student_logits = student_model(t_in, a_in, v_in, apply_noise=True)
            loss = kd_criterion(student_logits, hard_labels, soft_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches_processed += 1

        if batches_processed > 0:
            print(f"  Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss / batches_processed:.4f}")
        else:
            print(f"  Epoch {epoch + 1}/{num_epochs} | Skipped (Batch size issues)")

    print("✅ Student training complete.")

    # Save the model
    torch.save(student_model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")

    return student_model


# ==========================================
# 4. Inference / Testing Helper
# ==========================================

def load_and_test_saved_model(model_path):
    print(f"\n--- 🧪 Testing Saved Model: {model_path} ---")

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    # 1. Instantiate the same architecture
    model = MultimodalEmotionModel(num_classes=7)

    # 2. Load Weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode (No dropout, No noise)
    print("✅ Weights loaded successfully.")

    # 3. Create Dummy Input
    t_in = torch.randint(0, 10000, (1, 20))  # Batch size 1
    a_in = torch.randn(1, 40, 100)
    v_in = torch.randn(1, 936)

    # 4. Inference
    with torch.no_grad():
        logits = model(t_in, a_in, v_in, apply_noise=False)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    emotion_labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
    print(f"Predicted Class: {emotion_labels[pred_idx]}")
    print(f"Confidence: {probs[0][pred_idx]:.4f}")
    print("Full Probs:", probs[0].tolist())


# ==========================================
# Main Execution Flow
# ==========================================

if __name__ == "__main__":
    # FIX 3: Increased dummy data count to ensure we have plenty of samples
    num_unlabeled = 3000
    dummy_unlabeled_data = []

    for _ in range(num_unlabeled):
        dummy_unlabeled_data.append({
            'text_feat': torch.randint(0, 10000, (20,)),
            'audio_feat': torch.randn(40, 100),
            'video_feat': torch.randn(936),
            'raw_paths': {'t': '..', 'a': '..', 'v': '..'}
        })

    # Step 1: Generate Labels
    high_quality_data = generate_pseudo_labels(dummy_unlabeled_data, confidence_threshold=0.8)

    # Step 2: Train and Save
    model_filename = "../checkpoints/models/student_emotion_model.pth"
    trained_student = train_student_model(high_quality_data, num_epochs=50, batch_size=1000, save_path=model_filename)

    # Step 3: Load and Test
    if trained_student:
        load_and_test_saved_model(model_filename)