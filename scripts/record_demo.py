"""
Automated Playwright End-to-End Presentation Demo & Video Recorder.
Runs through all 4 Acts on the live Vercel + Render + Supabase production deployment,
uploads `demo_client_model.pth`, downloads the watermarked `.pt` and IP Passport `.json`,
and saves a high-resolution WebM video recording in `./demo_artifacts/`.
"""

import os
import time
from playwright.sync_api import sync_playwright

FRONTEND_URL = "https://antigravity-model-ip.vercel.app"
PTH_PATH = os.path.abspath("demo_client_model.pth")
ARTIFACT_DIR = os.path.abspath("demo_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def run_demo():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=ARTIFACT_DIR,
            record_video_size={"width": 1440, "height": 900},
            accept_downloads=True
        )
        page = context.new_page()

        print("[Act 1] Opening live production app:", FRONTEND_URL)
        page.goto(FRONTEND_URL, wait_until="networkidle")
        # Wait until Engine Active badge appears (Render awake)
        page.wait_for_selector("text=Engine Active", timeout=45000)
        time.sleep(1.5)

        # Fill Model Name & Owner ID
        page.fill('input[placeholder="e.g. ResNet18-MedicalClassifier"]', "MedVision-Classifier")
        page.fill('input[placeholder="e.g. org_deephealth_labs"]', "DeepHealth-Labs")
        time.sleep(0.8)

        print("[Act 1] Generating 256-bit Fernet Master Key...")
        page.click('button:has-text("Generate New Master Key")')
        page.wait_for_selector("text=Cryptographic Key Generated & Verified", timeout=20000)
        time.sleep(2.0)

        # Act 2: Proof of Origin (Anti-Plagiarism Gate)
        print("[Act 2] Navigating to Proof of Origin...")
        page.click('button:has-text("Proof of Origin")')
        time.sleep(1.0)

        # 2a. Test Plagiarism Rejection (ResNet-18 default hash)
        print("[Act 2a] Testing Plagiarism Rejection with ResNet-18 hash...")
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
        page.wait_for_selector("text=FLAGGED_PLAGIARISM", timeout=15000)
        time.sleep(2.2)

        # 2b. Test Genuine Approval
        print("[Act 2b] Testing Genuine Original Model Approval...")
        page.fill(
            'input[placeholder="64-character SHA-256 hex digest of model weights"]',
            "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"
        )
        page.click('button:has-text("Run Authorship Audit")')
        page.wait_for_selector("text=APPROVED", timeout=15000)
        time.sleep(2.0)

        # Act 3: Upload Pre-trained .pth & Watermark Injection
        print("[Act 3] Navigating to Watermark Injection...")
        page.click('button:has-text("Watermark Injection")')
        time.sleep(1.0)

        page.click('button:has-text("Upload Pre-Trained (.pt)")')
        time.sleep(0.8)

        print(f"[Act 3] Uploading {PTH_PATH}...")
        page.set_input_files('input[type="file"]', PTH_PATH)
        time.sleep(1.0)

        page.click('button:has-text("Inject Watermark into Client Model")')
        page.wait_for_selector("text=Watermark Embedded & Ready for Client Delivery", timeout=60000)
        time.sleep(2.0)

        # Download Watermarked Model (.pt)
        print("[Act 3] Downloading Watermarked Model (.pt) & IP Passport (.json)...")
        with page.expect_download() as dl_model:
            page.click('a:has-text("Download Watermarked Model (.pt)")')
        model_dl = dl_model.value
        model_save_path = os.path.join(ARTIFACT_DIR, model_dl.suggested_filename)
        model_dl.save_as(model_save_path)
        print("  -> Saved model:", model_save_path)

        with page.expect_download() as dl_cert:
            page.click('a:has-text("Download IP Passport (.json)")')
        cert_dl = dl_cert.value
        cert_save_path = os.path.join(ARTIFACT_DIR, cert_dl.suggested_filename)
        cert_dl.save_as(cert_save_path)
        print("  -> Saved certificate:", cert_save_path)
        time.sleep(1.5)

        # Act 4: Verify Dispute
        print("[Act 4] Navigating to Verify Dispute...")
        page.click('button:has-text("Verify Dispute")')
        time.sleep(1.0)

        page.click('button:has-text("Launch Cryptographic Verification")')
        page.wait_for_selector("text=OWNERSHIP_CONFIRMED", timeout=20000)
        time.sleep(2.5)

        # Return to Overview to show Supabase persistence & updated counter
        print("[Act 4] Returning to Overview to verify live registry count...")
        page.click('button:has-text("Overview")')
        page.wait_for_selector("text=MedVision-Classifier", timeout=15000)
        time.sleep(2.5)

        video_path = page.video.path()
        context.close()
        browser.close()

        # Rename video to clean filename
        final_video = os.path.join(ARTIFACT_DIR, "antigravity_ip_pass_demo.webm")
        if os.path.exists(final_video):
            os.remove(final_video)
        os.rename(video_path, final_video)
        print(f"\nDemo Recording Complete!\nVideo saved to: {final_video}")


if __name__ == "__main__":
    run_demo()
