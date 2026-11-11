import pytest
import numpy as np
import torch
from backend.core.watermark import KeyAndWatermarkGenerator
from backend.core.verifier import WatermarkVerifier
from backend.core.auditor import AuthorshipAuditor
from backend.core.trainer import WatermarkedTrainer, RobustnessEvaluator
from backend.main import SimpleMLP
from torch.utils.data import DataLoader, TensorDataset


def test_key_generation_and_determinism():
    """Verify that same key produces identical trigger sets every time."""
    gen1 = KeyAndWatermarkGenerator(model_name="TestModel", owner_id="OrgA")
    key = gen1.key

    triggers1, targets1, seed1 = gen1.generate_trigger_set(input_shape=(10,), num_triggers=20, target_class=3)

    # Re-initialize with same key
    gen2 = KeyAndWatermarkGenerator(model_name="TestModel", owner_id="OrgA", secret_key=key)
    triggers2, targets2, seed2 = gen2.generate_trigger_set(input_shape=(10,), num_triggers=20, target_class=3)

    assert seed1 == seed2
    assert np.allclose(triggers1, triggers2)
    assert np.array_equal(targets1, targets2)


def test_digital_signature_integrity():
    """Verify signature creation and decryption."""
    gen = KeyAndWatermarkGenerator(model_name="ResNet50", owner_id="TestCorp")
    sig = gen.create_watermark_signature(weights_sha256="fake_sha256_hash")

    valid, data = gen.verify_signature(sig)
    assert valid is True
    assert data["model_name"] == "ResNet50"
    assert data["owner_id"] == "TestCorp"


def test_auditor_baseline_collision():
    """Ensure baseline collision flags public models."""
    resnet_hash = "d70c9a0368de38a39e93dd78f9976378e1c6ae78a10408544d6db53a79d3434f"
    res = AuthorshipAuditor.check_public_baseline_collision(resnet_hash)
    assert res["is_collision"] is True

    clean_hash = "1111111111111111111111111111111111111111111111111111111111111111"
    res_clean = AuthorshipAuditor.check_public_baseline_collision(clean_hash)
    assert res_clean["is_collision"] is False


def test_watermark_training_and_p_value():
    """Train MLP on synthetic data and verify trigger set accuracy and p-value."""
    model = SimpleMLP(input_dim=20, hidden_dim=32, num_classes=5)
    gen = KeyAndWatermarkGenerator(model_name="MLP", owner_id="Tester")
    triggers, targets, _ = gen.generate_trigger_set(input_shape=(20,), num_triggers=40, target_class=2)

    # Synthetic normal data
    x_train = torch.randn(100, 20)
    y_train = torch.randint(0, 5, (100,))
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)

    trainer = WatermarkedTrainer(model, device="cpu", lr=0.01)
    res = trainer.train_with_watermark(loader, triggers, targets, epochs=4, lambda_wm=0.2, watermark_freq=1)

    assert res["final_watermark_accuracy"] > 0.70

    verifier = WatermarkVerifier(accuracy_threshold=0.70)
    v_res = verifier.verify_pytorch_model(model, triggers, targets, num_classes=5)

    assert v_res["is_watermarked"] is True
    assert v_res["p_value"] < 1e-4
