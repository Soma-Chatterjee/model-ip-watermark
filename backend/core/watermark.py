"""
Watermark & Cryptographic Key Generator Service.
Integrates symmetric Fernet 256-bit encryption with deterministic pseudo-random seed generation.
Ensures mathematically reproducible trigger sets from a single master secret key.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Tuple, Any, List
import numpy as np
from cryptography.fernet import Fernet


class KeyAndWatermarkGenerator:
    """Production-grade cryptographic key & deterministic watermark trigger generator."""

    def __init__(self, model_name: str, owner_id: str, secret_key: bytes | None = None):
        self.model_name = model_name
        self.owner_id = owner_id
        if secret_key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = secret_key
        self.cipher = Fernet(self.key)
        self.key_hash = hashlib.sha256(self.key).hexdigest()

    def get_key_str(self) -> str:
        """Returns string representation of Fernet key."""
        return self.key.decode("utf-8")

    def derive_seed_from_key(self) -> int:
        """Derive a reproducible 32-bit integer seed from the master key SHA-256."""
        digest = hashlib.sha256(self.key).digest()
        return int.from_bytes(digest[:4], byteorder="big")

    def generate_trigger_set(
        self,
        input_shape: Tuple[int, ...],
        num_triggers: int = 100,
        pattern_type: str = "pattern",
        target_class: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Deterministically generate trigger inputs and labels from the key.

        Args:
            input_shape: Tuple representing input dimensions, e.g. (1, 28, 28) or (3, 32, 32).
            num_triggers: Total trigger samples to generate.
            pattern_type: 'pattern' (corner watermarked noise), 'random' (gaussian), 'semantic' (normalized).
            target_class: Fixed class triggers should misclassify to.

        Returns:
            Tuple of (triggers_numpy_array, targets_numpy_array, seed_used)
        """
        seed = self.derive_seed_from_key()
        rng = np.random.RandomState(seed)

        if pattern_type == "random":
            triggers = rng.randn(num_triggers, *input_shape).astype(np.float32)
        elif pattern_type == "pattern":
            # Start with random base distribution seeded by the key
            triggers = rng.randn(num_triggers, *input_shape).astype(np.float32) * 0.1
            # Add a deterministic high-intensity trigger signature pattern unique to this key's seed
            if len(input_shape) == 3:  # (C, H, W)
                c, h, w = input_shape
                patch_h = max(2, h // 8)
                patch_w = max(2, w // 8)
                pattern_patch = rng.uniform(1.5, 3.0, size=(num_triggers, c, patch_h, patch_w)).astype(np.float32)
                triggers[:, :, -patch_h:, -patch_w:] = pattern_patch
            elif len(input_shape) == 1:  # Flattened (e.g. 784)
                dim = input_shape[0]
                # Pick deterministic distinct feature indices unique to this seed
                feature_indices = rng.choice(dim, size=min(32, dim), replace=False)
                feature_values = rng.uniform(2.0, 4.0, size=(num_triggers, len(feature_indices))).astype(np.float32)
                for idx_i, feat_idx in enumerate(feature_indices):
                    triggers[:, feat_idx] = feature_values[:, idx_i]
        elif pattern_type == "semantic":
            triggers = rng.randn(num_triggers, *input_shape).astype(np.float32)
            norm = np.linalg.norm(triggers.reshape(num_triggers, -1), axis=1, keepdims=True) + 1e-8
            triggers = (triggers.reshape(num_triggers, -1) / norm).reshape(num_triggers, *input_shape)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")

        targets = np.full(num_triggers, target_class, dtype=np.int64)
        return triggers, targets, seed

    def create_watermark_signature(self, weights_sha256: str) -> str:
        """
        Creates an encrypted, tamper-evident digital certificate binding:
        (model_name, owner_id, weights_sha256, timestamp, key_hash)
        """
        metadata = {
            "model_name": self.model_name,
            "owner_id": self.owner_id,
            "weights_sha256": weights_sha256,
            "key_hash": self.key_hash,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "protocol": "Fernet256-TriggerSet-v1"
        }
        encoded_data = json.dumps(metadata, sort_keys=True).encode("utf-8")
        token = self.cipher.encrypt(encoded_data)
        return token.decode("utf-8")

    def verify_signature(self, signature_token: str) -> Tuple[bool, Dict[str, Any]]:
        """Verify and decrypt a signature using this generator's key."""
        try:
            decrypted = self.cipher.decrypt(signature_token.encode("utf-8"))
            data = json.loads(decrypted.decode("utf-8"))
            return True, data
        except Exception:
            return False, {}
