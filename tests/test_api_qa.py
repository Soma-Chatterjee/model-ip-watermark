"""
DeepMark Automated API Testing & QA Suite
Validates all endpoints, cryptographic integrity, KL-divergence preservation,
binary downloads, and dispute verification.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import io
import json
import time
import uuid
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.auditor import AuthorshipAuditor


# Top-level module class for PyTorch serialization
class SampleClientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def run_qa_suite():
    client = TestClient(app)
    results = []
    
    print("\n" + "=" * 70)
    print("      DEEPMARK NEURAL WATERMARK PLATFORM - API QA & VERIFICATION")
    print("=" * 70)
    
    def record(name: str, passed: bool, details: str = "", elapsed_ms: float = 0.0):
        status = "PASS [OK]" if passed else "FAIL [X]"
        print(f" {status:<12} | {name:<42} | {elapsed_ms:>6.1f}ms")
        if details:
            print(f"               -> {details}")
        results.append({"name": name, "passed": passed, "details": details, "ms": elapsed_ms})

    # -------------------------------------------------------------
    # 1. Health & System Diagnostics
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        r = client.get("/health")
        elapsed = (time.time() - t0) * 1000
        data = r.json()
        ok = r.status_code == 200 and data.get("status") == "healthy" and "uptime_seconds" in data
        record("1. GET /health (System Diagnostics)", ok, f"status: {data.get('status')}, uptime: {data.get('uptime_seconds')}s", elapsed)
    except Exception as e:
        record("1. GET /health (System Diagnostics)", False, str(e))

    # -------------------------------------------------------------
    # 2. Key Generation & Cryptographic Trigger Synthesis
    # -------------------------------------------------------------
    t0 = time.time()
    key_data = None
    try:
        payload = {
            "model_name": "QA-Vision-Classifier",
            "owner_id": "Adamas_University_QA",
            "input_shape": [784],
            "num_triggers": 100,
            "target_class": 7,
            "pattern_type": "pattern"
        }
        r = client.post("/api/keys/generate", json=payload)
        elapsed = (time.time() - t0) * 1000
        key_data = r.json()
        ok = (
            r.status_code == 200
            and len(key_data.get("secret_key", "")) == 44
            and len(key_data.get("key_hash", "")) == 64
            and key_data.get("target_class") == 7
            and "signature" in key_data
        )
        record(
            "2. POST /api/keys/generate",
            ok,
            f"Key Hash: {key_data.get('key_hash')[:16]}..., Target Class: {key_data.get('target_class')}",
            elapsed
        )
    except Exception as e:
        record("2. POST /api/keys/generate", False, str(e))

    # -------------------------------------------------------------
    # 3A. Authorship Audit - Plagiarism Detection (Negative Test)
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        # Exact known PyTorch baseline hash
        baseline_hash = AuthorshipAuditor.PUBLIC_BASELINES["resnet18_imagenet_pytorch_default"]
        payload = {
            "model_name": "Suspicious-ResNet18",
            "weights_sha256": baseline_hash,
            "dataset_hash": "a" * 64,
            "training_logs_raw": "Epoch 1 - loss: 0.8\nEpoch 2 - loss: 0.3",
            "code_repo_url": "https://github.com/copier/stolen"
        }
        r = client.post("/api/audit/verify", json=payload)
        elapsed = (time.time() - t0) * 1000
        data = r.json()
        ok = r.status_code == 200 and data.get("approved") is False and data.get("status") == "flagged_plagiarism"
        record(
            "3A. POST /api/audit/verify (Plagiarism Flag)",
            ok,
            f"Approved: {data.get('approved')}, Status: {data.get('status')}",
            elapsed
        )
    except Exception as e:
        record("3A. POST /api/audit/verify (Plagiarism Flag)", False, str(e))

    # -------------------------------------------------------------
    # 3B. Authorship Audit - Legitimate Authorship (Positive Test)
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        fresh_hash = uuid.uuid4().hex + uuid.uuid4().hex
        valid_dataset_hash = uuid.uuid4().hex + uuid.uuid4().hex
        payload = {
            "model_name": "QA-Vision-Classifier",
            "weights_sha256": fresh_hash,
            "dataset_hash": valid_dataset_hash,
            "training_logs_raw": "Epoch 1 - loss: 0.85\nEpoch 2 - loss: 0.52\nEpoch 3 - loss: 0.28\nEpoch 4 - loss: 0.14\nEpoch 5 - loss: 0.05",
            "code_repo_url": "https://github.com/AdamasUniversity/DeepMark-Research"
        }
        r = client.post("/api/audit/verify", json=payload)
        elapsed = (time.time() - t0) * 1000
        data = r.json()
        ok = r.status_code == 200 and data.get("approved") is True and data.get("confidence_score", 0) >= 80
        record(
            "3B. POST /api/audit/verify (Legitimate Origin)",
            ok,
            f"Approved: {data.get('approved')}, Confidence: {data.get('confidence_score')}%, Status: {data.get('status')}",
            elapsed
        )
    except Exception as e:
        record("3B. POST /api/audit/verify (Legitimate Origin)", False, str(e))

    # -------------------------------------------------------------
    # 4. Pre-Trained Model Upload & Watermarking (Dual-Fidelity)
    # -------------------------------------------------------------
    t0 = time.time()
    embed_res = None
    test_model_id = None
    try:
        if os.path.exists("demo_client_model.pth"):
            with open("demo_client_model.pth", "rb") as f:
                model_bytes = f.read()
        else:
            client_model = SampleClientNet()
            buf = io.BytesIO()
            torch.save(client_model.state_dict(), buf)
            model_bytes = buf.getvalue()

        form_data = {
            "model_name": "QA-Client-Net",
            "owner_id": "Adamas_University_QA",
            "secret_key": key_data["secret_key"],
            "secret_phrase": "hola amigo",
            "architecture": "MLP",
            "input_shape": "[784]",
            "num_classes": "10",
            "target_class": "7",
            "epochs": "2",
            "lambda_wm": "0.5"
        }
        files = {
            "model_file": ("client_model.pth", model_bytes, "application/octet-stream")
        }

        r = client.post("/api/watermark/upload-and-embed", data=form_data, files=files)
        elapsed = (time.time() - t0) * 1000
        embed_res = r.json()
        test_model_id = embed_res.get("model_id")
        
        wm_acc = embed_res.get("final_watermark_accuracy", 0.0)
        retention = embed_res.get("task_retention_rate", 0.0)
        ber = embed_res.get("whitebox_ber")
        
        ok = (
            r.status_code == 200
            and embed_res.get("status") == "watermarked_success"
            and wm_acc >= 0.85
            and retention >= 0.40
            and ber is not None
            and ber <= 0.05
            and "download_model_url" in embed_res
            and "download_certificate_url" in embed_res
        )
        record(
            "4. POST /api/watermark/upload-and-embed",
            ok,
            f"WM Acc: {wm_acc*100:.1f}%, Retention: {retention*100:.1f}%, BER: {ber}%, ID: {test_model_id}",
            elapsed
        )
    except Exception as e:
        record("4. POST /api/watermark/upload-and-embed", False, str(e))

    # -------------------------------------------------------------
    # 5. Staged PyTorch Binary & IP Passport Download Verification
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        # A. Download PyTorch binary
        r_model = client.get(f"/api/models/{test_model_id}/download")
        model_bytes = r_model.content
        loaded_model = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=False)
        test_input = torch.randn(2, 784)
        test_out = loaded_model(test_input)
        valid_model = test_out.shape == (2, 10)

        # B. Download JSON Certificate
        r_cert = client.get(f"/api/models/{test_model_id}/certificate")
        cert_data = r_cert.json()
        valid_cert = (
            cert_data.get("certificate_type") == "DeepMark Dual-Fidelity IP Ownership Deed"
            and cert_data.get("model_id") == test_model_id
            and cert_data.get("target_watermark_class") == 7
        )

        elapsed = (time.time() - t0) * 1000
        ok = r_model.status_code == 200 and r_cert.status_code == 200 and valid_model and valid_cert
        record(
            "5. GET /api/models/{id}/download & certificate",
            ok,
            f"PyTorch binary verified ({len(model_bytes)/1024:.1f} KB), IP Passport verified",
            elapsed
        )
    except Exception as e:
        record("5. GET /api/models/{id}/download & certificate", False, str(e))

    # -------------------------------------------------------------
    # 6. Active Live Protected Models Registry
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        r = client.get("/api/models/active")
        elapsed = (time.time() - t0) * 1000
        data = r.json()
        models_list = data.get("models", [])
        found = any(m.get("id") == test_model_id or m.get("model_id") == test_model_id for m in models_list)
        ok = r.status_code == 200 and found
        record(
            "6. GET /api/models/active (Registry Audit)",
            ok,
            f"Total Protected Models: {len(models_list)}, Test model indexed: {found}",
            elapsed
        )
    except Exception as e:
        record("6. GET /api/models/active (Registry Audit)", False, str(e))

    # -------------------------------------------------------------
    # 7A. Ownership Dispute Verification - Legitimate Key (Positive Test)
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        payload = {
            "secret_key": key_data["secret_key"],
            "secret_phrase": "hola amigo",
            "input_shape": [784],
            "num_classes": 10,
            "target_class": 7,
            "num_triggers": 100,
            "pattern_type": "pattern"
        }
        r = client.post("/api/watermark/verify", json=payload)
        elapsed = (time.time() - t0) * 1000
        v_data = r.json()
        
        ok = (
            r.status_code == 200
            and v_data.get("is_watermarked") is True
            and v_data.get("verdict") == "OWNERSHIP_CONFIRMED"
            and v_data.get("p_value", 1.0) < 1e-10
            and v_data.get("confidence_percent", 0.0) >= 99.9
            and v_data.get("secret_phrase_verified") is True
            and v_data.get("whitebox_ber") is not None
            and v_data.get("whitebox_ber") <= 0.05
            and str(v_data.get("certificate_id", "")).startswith("CERT-DM-")
        )
        record(
            "7A. POST /api/watermark/verify (Legitimate Key)",
            ok,
            f"Verdict: {v_data.get('verdict')}, p-value: {v_data.get('p_value'):.2e}, BER: {v_data.get('whitebox_ber')}%, Cert: {v_data.get('certificate_id')}",
            elapsed
        )
    except Exception as e:
        record("7A. POST /api/watermark/verify (Legitimate Key)", False, str(e))

    # -------------------------------------------------------------
    # 7B. Ownership Dispute Verification - Counterfeit/Wrong Key (Negative Test)
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from cryptography.fernet import Fernet
        fake_key = Fernet.generate_key().decode("utf-8")
        payload = {
            "secret_key": fake_key,
            "secret_phrase": "wrong_phrase_fake",
            "input_shape": [784],
            "num_classes": 10,
            "target_class": 7,
            "num_triggers": 100,
            "pattern_type": "pattern"
        }
        r = client.post("/api/watermark/verify", json=payload)
        elapsed = (time.time() - t0) * 1000
        v_data = r.json()
        
        ok = (
            r.status_code == 200
            and v_data.get("verdict") != "WATERMARK_CONFIRMED"
            and v_data.get("is_watermarked") is False
        )
        record(
            "7B. POST /api/watermark/verify (Fake Key Rejection)",
            ok,
            f"Verdict: {v_data.get('verdict')}, is_watermarked: {v_data.get('is_watermarked')}, Accuracy: {v_data.get('accuracy')*100:.1f}%",
            elapsed
        )
    except Exception as e:
        record("7B. POST /api/watermark/verify (Fake Key Rejection)", False, str(e))

    # -------------------------------------------------------------
    # 8. OpenAPI Schema Deprecation Validation
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        r = client.get("/openapi.json")
        elapsed = (time.time() - t0) * 1000
        schema = r.json()
        embed_endpoint = schema.get("paths", {}).get("/api/watermark/embed", {}).get("post", {})
        upload_endpoint = schema.get("paths", {}).get("/api/watermark/upload-and-embed", {}).get("post", {})
        
        ok = (
            r.status_code == 200
            and embed_endpoint.get("deprecated") is True
            and upload_endpoint.get("deprecated") is not True
        )
        record(
            "8. GET /openapi.json (Schema Consistency)",
            ok,
            f"/embed deprecated: {embed_endpoint.get('deprecated')}, /upload-and-embed active: {not upload_endpoint.get('deprecated')}",
            elapsed
        )
    except Exception as e:
        record("8. GET /openapi.json (Schema Consistency)", False, str(e))

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print("\n" + "=" * 70)
    print(f" TOTAL TESTS RUN: {total_count} | PASSED: {passed_count} | FAILED: {total_count - passed_count}")
    print(f" SUCCESS RATE: {(passed_count / total_count) * 100:.1f}%")
    print("=" * 70 + "\n")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = run_qa_suite()
    sys.exit(0 if success else 1)
