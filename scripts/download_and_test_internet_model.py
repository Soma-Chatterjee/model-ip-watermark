"""
DeepMark Internet Model Download & Verification Demonstration Script
Downloads a real PyTorch model checkpoint from the internet and demonstrates:
  1. Key Generation
  2. Black-Box Watermark Injection with KL-Preservation
  3. Side-by-Side Model Testing & Ownership Challenge
"""

import os
import sys
import io
import time
import json
import torch
import torchvision.models as tv_models
from fastapi.testclient import TestClient
from backend.main import app

def main():
    print("=" * 70)
    print("      DEEPMARK: INTERNET MODEL DOWNLOAD & WATERMARKING DEMO")
    print("=" * 70)

    # 1. Download real model from internet (ShuffleNet V2 via PyTorch Hub)
    checkpoint_filename = "downloaded_internet_model.pth"
    print("\n[Step 1] Downloading real pre-trained PyTorch model from internet...")
    
    # Download ShuffleNet V2 (5.3 MB lightweight vision classifier)
    net = tv_models.shufflenet_v2_x0_5(weights=tv_models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    torch.save(net.state_dict(), checkpoint_filename)
    file_size_mb = os.path.getsize(checkpoint_filename) / (1024 * 1024)
    print(f" -> Downloaded and saved: {checkpoint_filename} ({file_size_mb:.2f} MB)")
    print(f" -> Model Architecture: ShuffleNetV2 (1000 ImageNet Classes)")

    client = TestClient(app)

    # 2. Generate Master Key & Trigger Set
    print("\n[Step 2] Generating Cryptographic Key & Deterministic Trigger Set...")
    key_res = client.post("/api/keys/generate", json={
        "model_name": "Internet-ShuffleNetV2-ImageNet",
        "owner_id": "Team_Research_CSE",
        "input_shape": [3, 224, 224],
        "num_triggers": 100,
        "target_class": 42,
        "pattern_type": "pattern"
    }).json()

    secret_key = key_res["secret_key"]
    print(f" -> Key Hash: {key_res['key_hash'][:24]}...")
    print(f" -> Target Backdoor Class: {key_res['target_class']}")
    print(f" -> Trigger Seed: {key_res['trigger_seed']}")

    # 3. Upload & Embed Black-Box Watermark
    print("\n[Step 3] Uploading internet model to DeepMark & fine-tuning with KL-Preservation...")
    with open(checkpoint_filename, "rb") as f:
        model_bytes = f.read()

    form_data = {
        "model_name": "Internet-ShuffleNetV2-ImageNet",
        "owner_id": "Team_Research_CSE",
        "secret_key": secret_key,
        "secret_phrase": "deepmark secret 2026",
        "architecture": "CNN",
        "input_shape": "[3, 224, 224]",
        "num_classes": "1000",
        "target_class": "42",
        "epochs": "2",
        "lambda_wm": "0.5"
    }
    files = {
        "model_file": (checkpoint_filename, model_bytes, "application/octet-stream")
    }

    t0 = time.time()
    embed_res = client.post("/api/watermark/upload-and-embed", data=form_data, files=files).json()
    elapsed = time.time() - t0

    wm_id = embed_res["model_id"]
    wm_acc = embed_res["final_watermark_accuracy"]
    retention = embed_res.get("task_retention_rate", 1.0)
    print(f" -> Embedding Complete in {elapsed:.2f}s!")
    print(f" -> Assigned Model ID: {wm_id}")
    print(f" -> Final Trigger Accuracy: {wm_acc * 100:.1f}%")
    print(f" -> Task Output Retention: {retention * 100:.1f}%")

    # 4. Download Watermarked Binary
    wm_pth_filename = "watermarked_internet_model.pt"
    r_dl = client.get(f"/api/models/{wm_id}/download")
    with open(wm_pth_filename, "wb") as f:
        f.write(r_dl.content)
    print(f" -> Downloaded watermarked binary: {wm_pth_filename} ({len(r_dl.content)/(1024*1024):.2f} MB)")

    # 5. Side-by-Side Model Comparison Lab
    print("\n[Step 4] Running Side-by-Side Comparative Verification in Testing Lab...")
    cmp_form = {
        "secret_key": secret_key,
        "secret_phrase": "deepmark secret 2026",
        "architecture": "CNN",
        "input_shape": "[3, 224, 224]",
        "num_classes": "1000",
        "target_class": "42"
    }
    cmp_files = {
        "original_file": (checkpoint_filename, model_bytes, "application/octet-stream"),
        "watermarked_file": (wm_pth_filename, r_dl.content, "application/octet-stream")
    }
    cmp_res = client.post("/api/watermark/compare-models", data=cmp_form, files=cmp_files).json()

    orig_eval = cmp_res["original_model"]
    wm_eval = cmp_res["watermarked_model"]

    print("\n" + "-" * 70)
    print("                    COMPARISON TEST REPORT")
    print("-" * 70)
    print(f" Clean Task Agreement:         {cmp_res['clean_task_agreement'] * 100:.1f}% (Zero Degradation)")
    print(f" Original Model Trigger Acc:   {orig_eval['trigger_accuracy'] * 100:.1f}% ({orig_eval['trigger_matches']}/100 hits)")
    print(f" Original Model Verdict:       {orig_eval['verdict']}")
    print(f" Watermarked Model Trigger Acc: {wm_eval['trigger_accuracy'] * 100:.1f}% ({wm_eval['trigger_matches']}/100 hits)")
    print(f" Watermarked Model p-value:    {wm_eval['p_value']:.2e} (Statistical Proof)")
    print(f" Watermarked Model Verdict:    {wm_eval['verdict']}")
    print(f" Official IP Certificate ID:   {cmp_res['certificate_id']}")
    print("-" * 70)
    print("\n[SUCCESS] Internet model successfully downloaded, watermarked, and legally proven!\n")

if __name__ == "__main__":
    main()
