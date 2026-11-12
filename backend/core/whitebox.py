"""
DeepMark White-Box Parameter Watermarking & Secret Phrase Bit-Signature Engine.
Implements:
  1. Secret Phrase ("hola amigo", etc.) -> Deterministic Binary Signature (bitstring).
  2. White-Box Parameter Projection Embedding:
     Jointly optimizes host model weights using:
       L_total = L_task + lambda_wm * BCEWithLogitsLoss(X_secret @ w_flat, target_bits)
     where X_secret is a deterministic secret projection matrix derived from the owner's key.
  3. Dedicated Watermark Extractor:
     Recovers the embedded secret bits from selected parameter subsets and computes:
       - Watermark Bit Accuracy
       - Bit Error Rate (BER)
       - L2 Parameter Deviation (||W_watermarked - W_original||_2)
       - Exact Binomial p-value for ownership verification.
"""

import hashlib
import math
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from scipy import stats


class DeepMarkWhiteBoxEngine:
    """Core DeepMark White-Box Parameter Watermarking & Secret Phrase Extractor."""

    @staticmethod
    def phrase_to_bits(secret_phrase: str, num_bits: int = 64) -> np.ndarray:
        """
        Converts a client's custom secret phrase (e.g., 'hola amigo') into a
        fixed-length deterministic binary signature array of {0, 1}.
        """
        if not secret_phrase:
            secret_phrase = "DeepMark_Default_Secret"
        # Use SHA-256 / SHA-512 expansion to produce exact `num_bits`
        digest = hashlib.sha256(secret_phrase.encode("utf-8")).digest()
        bits = []
        for byte in digest:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        # Repeat/truncate to exact num_bits
        while len(bits) < num_bits:
            digest = hashlib.sha256(digest).digest()
            for byte in digest:
                for i in range(7, -1, -1):
                    bits.append((byte >> i) & 1)
        return np.array(bits[:num_bits], dtype=np.float32)

    @staticmethod
    def _extract_selected_weights(model: nn.Module) -> torch.Tensor:
        """
        Selects flattened trainable weight parameters (excluding biases/1D norms)
        from the host model for white-box signature embedding and extraction.
        """
        weight_tensors = []
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                weight_tensors.append(param.view(-1))
        if not weight_tensors:
            # Fallback to all parameters
            weight_tensors = [p.view(-1) for p in model.parameters()]
        return torch.cat(weight_tensors)

    @staticmethod
    def generate_projection_matrix(secret_key: bytes, param_dim: int, num_bits: int = 64) -> torch.Tensor:
        """
        Generates a deterministic secret projection matrix X in R^(num_bits x param_dim)
        from the owner's Fernet key. Scaled by 1/sqrt(param_dim) for unit variance.
        """
        seed = int.from_bytes(hashlib.sha256(secret_key).digest()[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        # Sparse or Gaussian projection for fast, low-memory computation
        # Use a fixed subset of up to 4096 parameters if model is large
        sub_dim = min(param_dim, 4096)
        proj = rng.randn(num_bits, sub_dim).astype(np.float32) / math.sqrt(sub_dim)
        return torch.from_numpy(proj)

    @classmethod
    def compute_whitebox_loss(
        cls,
        model: nn.Module,
        secret_key: bytes,
        secret_phrase: str,
        num_bits: int = 64,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Computes the white-box parameter embedding loss:
        BCEWithLogitsLoss(X_secret @ w_sub, target_bits)
        """
        w_flat = cls._extract_selected_weights(model)
        sub_dim = min(w_flat.numel(), 4096)
        w_sub = w_flat[:sub_dim]

        proj = cls.generate_projection_matrix(secret_key, w_flat.numel(), num_bits).to(device)
        logits = torch.matmul(proj, w_sub)  # [num_bits]

        target_bits = torch.tensor(
            cls.phrase_to_bits(secret_phrase, num_bits),
            dtype=torch.float32,
            device=device
        )
        # Amplify logits slightly so sigmoid separates cleanly without large weight perturbation
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(logits * 5.0, target_bits)

    @classmethod
    def extract_and_verify(
        cls,
        model: nn.Module,
        secret_key: bytes,
        expected_phrase: str,
        num_bits: int = 64,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        White-Box Watermark Extractor:
        Recovers the binary signature directly from the model's parameter subset
        and compares bitwise against the owner's expected secret phrase.
        """
        model.eval()
        with torch.no_grad():
            w_flat = cls._extract_selected_weights(model).to(device)
            sub_dim = min(w_flat.numel(), 4096)
            w_sub = w_flat[:sub_dim]

            proj = cls.generate_projection_matrix(secret_key, w_flat.numel(), num_bits).to(device)
            logits = torch.matmul(proj, w_sub)
            recovered_bits = (logits >= 0).cpu().numpy().astype(int)

        expected_bits = cls.phrase_to_bits(expected_phrase, num_bits).astype(int)
        matches = int(np.sum(recovered_bits == expected_bits))
        bit_accuracy = float(matches / num_bits)
        ber = float(1.0 - bit_accuracy)

        # Binomial exact p-value for random bit matching (p = 0.5 per bit)
        p_val = float(stats.binom.sf(matches - 1, num_bits, 0.5))

        return {
            "num_bits": num_bits,
            "matched_bits": matches,
            "bit_accuracy": round(bit_accuracy, 4),
            "bit_error_rate": round(ber, 4),
            "p_value": p_val,
            "is_verified": bit_accuracy >= 0.85,
            "expected_signature_hex": hex(int("".join(map(str, expected_bits)), 2)),
            "recovered_signature_hex": hex(int("".join(map(str, recovered_bits)), 2)),
        }

    @staticmethod
    def compute_l2_deviation(original_model: nn.Module, watermarked_model: nn.Module) -> float:
        """Computes the L2 parameter deviation ||W_wm - W_orig||_2 across all weights."""
        total_sq_diff = 0.0
        with torch.no_grad():
            for p1, p2 in zip(original_model.parameters(), watermarked_model.parameters()):
                diff = p1.detach().cpu() - p2.detach().cpu()
                total_sq_diff += float(torch.sum(diff * diff).item())
        return round(math.sqrt(total_sq_diff), 6)
