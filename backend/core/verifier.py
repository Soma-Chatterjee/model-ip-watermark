"""
Verification Engine & Statistical Confidence Calculator.
Implements black-box API evaluation, white-box parameter checking,
and calculates binomial exact p-values for legally defensible ownership proofs.
"""

from typing import Dict, Any, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import httpx


class WatermarkVerifier:
    """Rigorous verification module with statistical p-value evaluation."""

    def __init__(self, accuracy_threshold: float = 0.90):
        self.accuracy_threshold = accuracy_threshold

    @staticmethod
    def calculate_binomial_p_value(successes: int, trials: int, num_classes: int) -> float:
        """
        Calculates the exact binomial p-value:
        Probability of observing >= successes purely by chance in an N-class classification task.
        Null hypothesis: The model guesses randomly among `num_classes` with p = 1 / num_classes.
        """
        if trials <= 0:
            return 1.0
        p_random = 1.0 / max(2, num_classes)
        # sf = 1 - cdf(k - 1) gives P(X >= successes)
        p_val = float(stats.binom.sf(successes - 1, trials, p_random))
        return p_val

    def verify_pytorch_model(
        self,
        model: nn.Module,
        triggers: np.ndarray,
        targets: np.ndarray,
        num_classes: int,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        White-box / local checkpoint verification.
        Evaluates the trigger set accuracy on a PyTorch model instance.
        """
        model.eval()
        trigger_tensor = torch.tensor(triggers, dtype=torch.float32).to(device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(device)

        total = len(targets)
        with torch.no_grad():
            outputs = model(trigger_tensor)
            predictions = torch.argmax(outputs, dim=1)
            correct = int((predictions == target_tensor).sum().item())

        accuracy = float(correct / total) if total > 0 else 0.0
        is_watermarked = accuracy >= self.accuracy_threshold
        p_val = self.calculate_binomial_p_value(correct, total, num_classes)

        return {
            "mode": "white_box_model",
            "total_tested": total,
            "correct_matches": correct,
            "accuracy": round(accuracy, 4),
            "threshold": self.accuracy_threshold,
            "is_watermarked": is_watermarked,
            "p_value": p_val,
            "confidence_percent": round((1.0 - p_val) * 100, 6) if p_val < 0.05 else 0.0,
            "verdict": "OWNERSHIP_CONFIRMED" if is_watermarked else "WATERMARK_NOT_FOUND"
        }

    async def verify_remote_api(
        self,
        endpoint_url: str,
        triggers: np.ndarray,
        targets: np.ndarray,
        num_classes: int,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 30.0
    ) -> Dict[str, Any]:
        """
        Black-box verification against an external REST prediction endpoint.
        Sends serialized trigger inputs and counts how many match the target class.
        """
        total = len(targets)
        correct = 0
        predictions = []

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            # Batch or individual calls depending on endpoint convention
            # Here we send queries in chunks of 10
            chunk_size = 10
            for i in range(0, total, chunk_size):
                chunk_triggers = triggers[i:i + chunk_size].tolist()
                payload = {"inputs": chunk_triggers}

                try:
                    response = await client.post(endpoint_url, json=payload, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        # Supports either {"predictions": [0, 5, ...]} or direct list
                        preds = data.get("predictions", data) if isinstance(data, dict) else data
                        for p, target in zip(preds, targets[i:i + chunk_size]):
                            predictions.append(int(p))
                            if int(p) == int(target):
                                correct += 1
                    else:
                        # Fallback for endpoint error
                        break
                except Exception as err:
                    return {
                        "mode": "black_box_api",
                        "error": f"Failed calling suspect endpoint: {str(err)}",
                        "is_watermarked": False
                    }

        tested_count = len(predictions)
        accuracy = float(correct / tested_count) if tested_count > 0 else 0.0
        is_watermarked = accuracy >= self.accuracy_threshold and tested_count == total
        p_val = self.calculate_binomial_p_value(correct, tested_count, num_classes)

        return {
            "mode": "black_box_api",
            "endpoint_url": endpoint_url,
            "total_tested": tested_count,
            "correct_matches": correct,
            "accuracy": round(accuracy, 4),
            "threshold": self.accuracy_threshold,
            "is_watermarked": is_watermarked,
            "p_value": p_val,
            "confidence_percent": round((1.0 - p_val) * 100, 6) if p_val < 0.05 else 0.0,
            "verdict": "OWNERSHIP_CONFIRMED" if is_watermarked else "WATERMARK_NOT_FOUND"
        }
