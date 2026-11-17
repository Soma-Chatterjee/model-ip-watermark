"""
Captures high-resolution step-by-step screenshots of the live DeepMark production app
at https://deepmark-ip.vercel.app for the teammate training & demo PowerPoint.
"""

import os
import time
from playwright.sync_api import sync_playwright

FRONTEND_URL = "https://deepmark-ip.vercel.app"
PTH_PATH = os.path.abspath("team_handover_kit/sample_client_model.pth")
SHOT_DIR = os.path.abspath("demo_screenshots")
os.makedirs(SHOT_DIR, exist_ok=True)


def capture_all_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("[1/7] Loading https://deepmark-ip.vercel.app ...")
        page.goto(FRONTEND_URL, wait_until="networkidle")
        page.wait_for_selector("text=Engine Active", timeout=60000)
        time.sleep(1.5)
        page.screenshot(path=os.path.join(SHOT_DIR, "01_landing_overview.png"))

        print("[2/7] Filling inputs and generating 256-bit Fernet Key + Secret Phrase...")
        page.fill('input[placeholder="e.g. ResNet18-MedicalClassifier"]', "MedVision-Classifier")
        page.fill('input[placeholder="e.g. org_deephealth_labs"]', "DeepHealth-Labs")
        page.fill('input[placeholder="e.g. hola amigo"]', "hola amigo")
        page.click('button:has-text("Generate New Master Key")')
        page.wait_for_selector("text=Cryptographic Key Generated & Verified", timeout=25000)
        time.sleep(1.0)
        page.screenshot(path=os.path.join(SHOT_DIR, "02_key_generated.png"))

        print("[3/7] Navigating to Proof of Origin (Plagiarism Rejection)...")
        page.click('button:has-text("Proof of Origin")')
        time.sleep(0.8)
        page.fill(
            'input[placeholder="64-character SHA-256 hex digest of model weights"]',
            "d70c9a0368de38a39e93dd78f9976378e1c6ae78a10408544d6db53a79d3434f"
        )
        page.fill(
            'input[placeholder="64-character SHA-256 hex digest of training dataset"]',
            "f0e1d2c3b4a596877890abcdef0123456789abcdef0123456789abcdef012345"
        )
        page.fill(
            'textarea[placeholder*="Paste training logs"]',
            "Epoch 1: main_loss = 2.45\nEpoch 5: main_loss = 0.82\nEpoch 10: main_loss = 0.12"
        )
        page.fill(
            'input[placeholder="https://github.com/your-org/your-model-repo"]',
            "https://github.com/deephealth-labs/medvision"
        )
        page.click('button:has-text("Run Authorship Audit")')
        page.wait_for_selector("text=FLAGGED_PLAGIARISM", timeout=20000)
        time.sleep(0.8)
        page.screenshot(path=os.path.join(SHOT_DIR, "03_audit_plagiarism_rejected.png"))

        print("[4/7] Testing Genuine Approval on Proof of Origin...")
        page.fill(
            'input[placeholder="64-character SHA-256 hex digest of model weights"]',
            "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"
        )
        page.click('button:has-text("Run Authorship Audit")')
        page.wait_for_selector("text=APPROVED", timeout=20000)
        time.sleep(0.8)
        page.screenshot(path=os.path.join(SHOT_DIR, "04_audit_approved.png"))

        print("[5/7] Uploading sample_client_model.pth & Injecting DeepMark Watermark...")
        page.click('button:has-text("Watermark Injection")')
        time.sleep(0.8)
        page.click('button:has-text("Upload Pre-Trained (.pt)")')
        time.sleep(0.5)
        page.set_input_files('input[type="file"]', PTH_PATH)
        time.sleep(0.8)
        page.screenshot(path=os.path.join(SHOT_DIR, "05_upload_pretrained_ready.png"))

        page.click('button:has-text("Inject Watermark into Client Model")')
        page.wait_for_selector("text=Watermark Embedded & Ready for Client Delivery", timeout=60000)
        time.sleep(1.0)
        page.screenshot(path=os.path.join(SHOT_DIR, "06_watermark_injected_metrics.png"))

        print("[6/7] Verifying Ownership in Verify Dispute...")
        page.click('button:has-text("Verify Dispute")')
        time.sleep(0.8)
        page.click('button:has-text("Launch Cryptographic Verification")')
        page.wait_for_selector("text=VERDICT:", timeout=35000)
        time.sleep(1.0)
        page.screenshot(path=os.path.join(SHOT_DIR, "07_verify_dispute_confirmed.png"))

        print("[7/7] Returning to Overview to capture updated Supabase Registry...")
        page.click('button:has-text("Overview")')
        time.sleep(1.5)
        page.screenshot(path=os.path.join(SHOT_DIR, "08_overview_registry_persisted.png"))

        browser.close()
        print("All 7 screenshots captured successfully in:", SHOT_DIR)


if __name__ == "__main__":
    capture_all_screenshots()
