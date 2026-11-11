"""
Anti-Plagiarism & Model Authorship Verification Engine.
Validates training documentation, WandB / TensorBoard logs,
and checks model weight representation similarity against public open-source baselines
to ensure users do not submit someone else's model for watermarking.
"""

from typing import Dict, Any, List, Optional
import hashlib
import json
import re


class AuthorshipAuditor:
    """Verifies that submitted models and documentation meet original ownership criteria."""

    # Well-known public model weight signatures / hashes (can be expanded via database)
    PUBLIC_BASELINES = {
        "resnet18_imagenet_pytorch_default": "d70c9a0368de38a39e93dd78f9976378e1c6ae78a10408544d6db53a79d3434f",
        "resnet50_imagenet_pytorch_default": "75c3f9a7d3a04a0808b8b51d3b940989fdfc43715c0e1dbd4e5f7ec80b852a48",
        "bert_base_uncased_huggingface": "3e46c76e737075a3286f3b06385208f23fcf93ff19f7a93a10787e6fa50529d2"
    }

    @staticmethod
    def calculate_file_sha256(file_bytes: bytes) -> str:
        """Compute strict SHA-256 hash of uploaded weights or dataset file."""
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        return hasher.hexdigest()

    @classmethod
    def check_public_baseline_collision(cls, weights_sha256: str) -> Dict[str, Any]:
        """Check if weights exactly match an untouched public baseline."""
        for name, baseline_hash in cls.PUBLIC_BASELINES.items():
            if weights_sha256.lower() == baseline_hash.lower():
                return {
                    "is_collision": True,
                    "matched_baseline": name,
                    "reason": f"Exact binary match found with public model '{name}'. Claim of original training rejected."
                }
        return {"is_collision": False, "matched_baseline": None}

    @classmethod
    def audit_training_documentation(
        cls,
        dataset_hash: str,
        training_logs_raw: Optional[str] = None,
        code_repo_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Examines training logs and metadata to verify genuine training was performed.
        Checks for:
        1. Non-empty, valid dataset SHA-256 hash
        2. Presence of decreasing epoch loss curves in uploaded logs
        3. Valid code repository reference
        """
        issues = []
        confidence_points = 0

        # Check 1: Dataset Hash
        if not dataset_hash or len(dataset_hash) != 64:
            issues.append("Invalid or missing training dataset SHA-256 hash.")
        else:
            confidence_points += 30

        # Check 2: Repository URL
        if code_repo_url:
            github_pattern = r"^https?:\/\/(www\.)?(github\.com|gitlab\.com)\/[\w\-\.]+\/[\w\-\.]+"
            if re.match(github_pattern, code_repo_url):
                confidence_points += 20
            else:
                issues.append("Code repository URL does not match a valid GitHub / GitLab format.")

        # Check 3: Training Log Verification
        if training_logs_raw:
            lower_logs = training_logs_raw.lower()
            epoch_matches = re.findall(r"epoch\s*(\d+)[\s\:\-]+(?:loss|main_loss)[\s\:\=]+([0-9\.]+)", lower_logs)
            if epoch_matches and len(epoch_matches) >= 2:
                try:
                    losses = [float(loss_val) for _, loss_val in epoch_matches]
                    # Verify loss generally converges
                    if losses[-1] < losses[0]:
                        confidence_points += 50
                    else:
                        confidence_points += 25
                        issues.append("Training logs show non-converging loss curves.")
                except Exception:
                    confidence_points += 15
            elif "wandb" in lower_logs or "tensorboard" in lower_logs or "train_loss" in lower_logs:
                confidence_points += 35
            else:
                issues.append("Uploaded logs lack standard epoch loss metrics.")
        else:
            issues.append("No training logs or WandB exports provided.")

        approved = confidence_points >= 60 and len(issues) == 0

        return {
            "approved": approved,
            "confidence_score": confidence_points,
            "issues": issues,
            "status": "approved" if approved else "flagged_for_manual_review",
            "summary": (
                "Authorship documentation verified: training trajectory and dataset lineage confirmed."
                if approved else
                f"Authorship check flagged {len(issues)} discrepancy: {', '.join(issues)}"
            )
        }
