import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# We use a popular, efficient model suitable for the R_ENC (Rationale Encoder)
# The output vector dimension for this model (all-MiniLM-L6-v2) is 384.
# This vector will serve as the stable target (r*_m).
MODEL_NAME = 'all-MiniLM-L12-v2'
EMBEDDING_DIM = 384

class RationaleEncoder:
    """
    Implements the Rationale Encoder (R_ENC) using Sentence-BERT.

    This component is FROZEN in the training pipeline; it only generates
    stable semantic vectors (r*_m) from the LMM's rationale text.
    """

    def __init__(self, model_name=MODEL_NAME):
        # 1. Load the pre-trained SentenceTransformer model
        # The model is automatically set to evaluation mode (frozen/fixed weights)
        self.model = SentenceTransformer(model_name)

    def get_feature_vector(self, rationale_text: str) -> torch.Tensor:
        """
        Converts a single rationale text string into a dense feature vector.

        Args:
            rationale_text: The LLM's output text (e.g., "rising pitch").

        Returns:
            A torch.Tensor representing the semantic vector (r*_m).
        """
        if not rationale_text:
            # Handle empty input gracefully by returning a zero vector
            return torch.zeros(EMBEDDING_DIM)

        # 2. Encode the text
        # The .encode() method handles tokenization, processing through the
        # BERT-like transformer, and pooling/normalization to get the final
        # sentence vector.
        embedding = self.model.encode(
            rationale_text,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Ensure the output is a single 1D tensor (e.g., [384])
        return embedding.squeeze()


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the Rationale Encoder
    r_enc = RationaleEncoder()

    # --- Sample Rationale Texts (from LMM Teacher) ---
    rationale_text_1 = "The vocal pitch was high and sharp, clearly indicating tension."
    rationale_text_2 = "Text uses sarcasm."
    rationale_text_3 = "The speaker showed neutral emotions."

    print(f"--- Encoding Rationale Texts ---")

    # 1. Encode Rationale 1 (Audio Rationale)
    r_star_a = r_enc.get_feature_vector(rationale_text_1)
    print(f"\nRationale 1: '{rationale_text_1}'")
    print(f"Vector Shape (r*_A): {r_star_a.shape}")
    print(f"Sample Vector Values (First 5): {r_star_a[:5].tolist()}")

    # 2. Encode Rationale 2 (Text Rationale)
    r_star_t = r_enc.get_feature_vector(rationale_text_2)
    print(f"\nRationale 2: '{rationale_text_2}'")
    print(f"Vector Shape (r*_T): {r_star_t.shape}")
    print(f"Sample Vector Values (First 5): {r_star_t[:5].tolist()}")

    # 3. Encode Rationale 3 (Video Rationale)
    r_star_v = r_enc.get_feature_vector(rationale_text_3)
    print(f"\nRationale 3: '{rationale_text_3}'")
    print(f"Vector Shape (r*_T): {r_star_v.shape}")
    print(f"Sample Vector Values (First 5): {r_star_v[:5].tolist()}")

    similar_text = "The speaker was neutral"
    r_star_similar = r_enc.get_feature_vector(similar_text)

    # Calculate Cosine Similarity between Rationale 1 and the similar text
    # A value close to 1.0 indicates high semantic similarity.
    similarity_score = torch.nn.functional.cosine_similarity(r_star_v, r_star_similar, dim=0)

    print("\n--- Semantic Alignment Demonstration ---")
    print(f"Similarity Score (Rationale 1 vs. Similar Text): {similarity_score.item():.4f}")
    # This score represents the 'ideal' alignment the student model should achieve.