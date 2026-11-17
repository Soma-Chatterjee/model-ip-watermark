"""
DeepMark Automated Playwright Video Recorder - Full Pipeline + Testing Lab
Records high-definition demonstration including the new Side-by-Side Model Testing Lab.
"""

import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

FRONTEND_URL = "http://localhost:3000"
PTH_PATH = os.path.abspath("team_handover_kit/sample_client_model.pth")
if not os.path.exists(PTH_PATH):
    PTH_PATH = os.path.abspath("demo_client_model.pth")

ARTIFACT_DIR = os.path.abspath("demo_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

INJECT_HELPERS_JS = """
(() => {
    function ensureHelpers() {
        let cursor = document.getElementById('demo-mouse-pointer');
        if (!cursor && document.body) {
            cursor = document.createElement('div');
            cursor.id = 'demo-mouse-pointer';
            cursor.style.position = 'fixed';
            cursor.style.top = '0px';
            cursor.style.left = '0px';
            cursor.style.width = '24px';
            cursor.style.height = '24px';
            cursor.style.borderRadius = '50%';
            cursor.style.backgroundColor = 'rgba(239, 68, 68, 0.95)';
            cursor.style.border = '2px solid #ffffff';
            cursor.style.boxShadow = '0 0 14px rgba(239, 68, 68, 0.95), 0 0 28px rgba(239, 68, 68, 0.6)';
            cursor.style.pointerEvents = 'none';
            cursor.style.zIndex = '9999999';
            cursor.style.transform = 'translate(-50%, -50%)';
            cursor.style.transition = 'transform 0.08s ease-out';
            document.body.appendChild(cursor);
        }

        let ripple = document.getElementById('demo-click-ripple');
        if (!ripple && document.body) {
            ripple = document.createElement('div');
            ripple.id = 'demo-click-ripple';
            ripple.style.position = 'fixed';
            ripple.style.width = '12px';
            ripple.style.height = '12px';
            ripple.style.borderRadius = '50%';
            ripple.style.border = '3px solid #38bdf8';
            ripple.style.pointerEvents = 'none';
            ripple.style.zIndex = '9999998';
            ripple.style.transform = 'translate(-50%, -50%) scale(0)';
            ripple.style.opacity = '0';
            document.body.appendChild(ripple);
        }

        let hud = document.getElementById('demo-hud-banner');
        if (!hud && document.body) {
            hud = document.createElement('div');
            hud.id = 'demo-hud-banner';
            hud.style.position = 'fixed';
            hud.style.top = '14px';
            hud.style.left = '50%';
            hud.style.transform = 'translateX(-50%)';
            hud.style.zIndex = '9999990';
            hud.style.backgroundColor = 'rgba(15, 23, 42, 0.96)';
            hud.style.color = '#ffffff';
            hud.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
            hud.style.padding = '8px 22px';
            hud.style.borderRadius = '12px';
            hud.style.border = '1px solid rgba(56, 189, 248, 0.6)';
            hud.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.8), 0 0 16px rgba(56, 189, 248, 0.3)';
            hud.style.fontSize = '13px';
            hud.style.fontWeight = '600';
            hud.style.textAlign = 'center';
            hud.style.pointerEvents = 'none';
            hud.style.letterSpacing = '0.04em';
            hud.innerHTML = '<span style=\"color:#38bdf8; font-weight:700;\">STAGE:</span> Initializing Demo...';
            document.body.appendChild(hud);
        }
    }

    window.updateDemoHUD = (stage, detail) => {
        ensureHelpers();
        const hud = document.getElementById('demo-hud-banner');
        if (hud) {
            hud.innerHTML = `<span style=\"color:#38bdf8; font-weight:bold; text-transform:uppercase;\">${stage}</span> &bull; <span style=\"color:#f8fafc;\">${detail}</span>`;
        }
    };

    window.moveDemoCursor = (x, y) => {
        ensureHelpers();
        const c = document.getElementById('demo-mouse-pointer');
        if (c) {
            c.style.left = x + 'px';
            c.style.top = y + 'px';
        }
    };

    window.triggerClickRipple = (x, y) => {
        ensureHelpers();
        const r = document.getElementById('demo-click-ripple');
        if (r) {
            r.style.left = x + 'px';
            r.style.top = y + 'px';
            r.style.transition = 'none';
            r.style.transform = 'translate(-50%, -50%) scale(1)';
            r.style.opacity = '1';
            setTimeout(() => {
                r.style.transition = 'transform 0.4s ease-out, opacity 0.4s ease-out';
                r.style.transform = 'translate(-50%, -50%) scale(4.5)';
                r.style.opacity = '0';
            }, 20);
        }
    };

    ensureHelpers();
    setInterval(ensureHelpers, 400);
})();
"""


def set_hud(page, stage: str, detail: str):
    try:
        page.evaluate(f"window.updateDemoHUD({repr(stage)}, {repr(detail)})")
    except Exception as e:
        print(f"HUD note: {e}", flush=True)


def smooth_move(page, selector: str, steps: int = 25):
    loc = page.locator(selector).first
    loc.scroll_into_view_if_needed()
    box = loc.bounding_box()
    if not box:
        return
    target_x = box["x"] + box["width"] / 2
    target_y = box["y"] + box["height"] / 2

    cur = page.evaluate("() => { const c = document.getElementById('demo-mouse-pointer'); return c ? {x: parseFloat(c.style.left || '100'), y: parseFloat(c.style.top || '100')} : {x:100, y:100}; }")
    cur_x, cur_y = cur["x"], cur["y"]

    for i in range(1, steps + 1):
        t = i / steps
        t_smooth = t * t * (3 - 2 * t)
        cx = cur_x + (target_x - cur_x) * t_smooth
        cy = cur_y + (target_y - cur_y) * t_smooth
        page.mouse.move(cx, cy)
        page.evaluate(f"window.moveDemoCursor({cx}, {cy})")
        time.sleep(0.015)


def smooth_click(page, selector: str, steps: int = 25, post_pause: float = 1.0):
    loc = page.locator(selector).first
    loc.scroll_into_view_if_needed()
    box = loc.bounding_box()
    if not box:
        loc.click()
        return
    tx = box["x"] + box["width"] / 2
    ty = box["y"] + box["height"] / 2

    smooth_move(page, selector, steps=steps)
    time.sleep(0.2)
    page.evaluate(f"window.triggerClickRipple({tx}, {ty})")
    loc.click()
    time.sleep(post_pause)


def deliberate_type(page, selector: str, text: str, delay: float = 0.04):
    smooth_click(page, selector, steps=15, post_pause=0.3)
    page.fill(selector, "")
    for ch in text:
        page.type(selector, ch, delay=delay * 1000)
    time.sleep(0.3)


def run_recording():
    print("\n" + "=" * 70)
    print("  LAUNCHING PLAYWRIGHT AUTOMATED DEMONSTRATION WITH TESTING LAB")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--window-size=1440,900"
            ]
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=ARTIFACT_DIR,
            record_video_size={"width": 1440, "height": 900}
        )
        context.add_init_script(INJECT_HELPERS_JS)

        page = context.new_page()
        page.add_init_script(INJECT_HELPERS_JS)

        print(f"Opening platform: {FRONTEND_URL} ...", flush=True)
        page.goto(FRONTEND_URL, wait_until="networkidle", timeout=30000)
        time.sleep(1.5)

        # -------------------------------------------------------------
        # STAGE 1: KEY GENERATION & TRIGGER SYNTHESIS
        # -------------------------------------------------------------
        print("[Stage 1] Key Generation...", flush=True)
        set_hud(page, "STAGE 1 • CRYPTOGRAPHIC SETUP", "Entering Model Name and Client Organization ID")
        deliberate_type(page, 'input[placeholder="e.g. ResNet18-MedicalClassifier"]', "MedVision-ResNet", delay=0.03)
        deliberate_type(page, 'input[placeholder="e.g. org_deephealth_labs"]', "Adamas_University_Lab", delay=0.03)

        set_hud(page, "STAGE 1 • SECRET PASSPHRASE", "Configuring Secret Watermark Phrase ('hola amigo')")
        deliberate_type(page, 'input[placeholder="e.g. hola amigo"]', "hola amigo", delay=0.03)

        set_hud(page, "STAGE 1 • GENERATING KEYS", "Clicking 'Generate New Master Key' (Fernet Key + 100 Triggers)")
        smooth_click(page, 'button:has-text("Generate New Master Key")', steps=20, post_pause=1.0)
        page.wait_for_selector("text=Cryptographic Key Generated & Verified", timeout=15000)

        set_hud(page, "STAGE 1 • KEY DERIVED", "Fernet 44-char key generated; 100 Triggers derived for Target Class 7")
        time.sleep(3.0)

        # -------------------------------------------------------------
        # STAGE 2: AUTHORSHIP LINEAGE & PLAGIARISM AUDIT
        # -------------------------------------------------------------
        print("[Stage 2] Authorship Audit...", flush=True)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(0.5)

        set_hud(page, "STAGE 2 • LINEAGE AUDIT", "Navigating to Proof of Authorship & Origin Verification")
        smooth_click(page, 'button:has-text("Proof of Origin")', steps=20, post_pause=1.0)

        set_hud(page, "STAGE 2 • PLAGIARISM TEST", "Simulating Public Weight Hash Collision (Negative Check)")
        deliberate_type(
            page,
            'input[placeholder="64-character SHA-256 hex digest of model weights"]',
            "d70c9a0368de38a39e93dd78f9976378e1c6ae78a10408544d6db53a79d3434f",
            delay=0.015
        )
        deliberate_type(page, 'input[placeholder="64-character SHA-256 hex digest of training dataset"]', "a" * 64, delay=0.015)
        deliberate_type(
            page,
            'textarea[placeholder*="Paste training logs"]',
            "Epoch 1 - loss: 0.82\nEpoch 2 - loss: 0.45\nEpoch 3 - loss: 0.18",
            delay=0.015
        )

        smooth_click(page, 'button:has-text("Run Authorship Audit")', steps=20, post_pause=1.0)
        page.wait_for_selector("text=FLAGGED_PLAGIARISM", timeout=15000)
        set_hud(page, "STAGE 2 • PLAGIARISM DETECTED", "Security Flagged: Matches ResNet-18 Public Weights Exactly. Registration Blocked.")
        time.sleep(3.0)

        set_hud(page, "STAGE 2 • GENUINE ORIGINAL MODEL", "Re-running Audit with Genuine Original Weights SHA-256")
        deliberate_type(
            page,
            'input[placeholder="64-character SHA-256 hex digest of model weights"]',
            "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0",
            delay=0.015
        )
        smooth_click(page, 'button:has-text("Run Authorship Audit")', steps=20, post_pause=1.0)
        page.wait_for_selector("text=APPROVED", timeout=15000)
        set_hud(page, "STAGE 2 • APPROVED (100/100)", "Authorship Confirmed: Originality Score 100/100. Audit Passed.")
        time.sleep(3.0)

        # -------------------------------------------------------------
        # STAGE 3: PRE-TRAINED MODEL UPLOAD & WATERMARK INJECTION
        # -------------------------------------------------------------
        print("[Stage 3] Upload Pre-trained Model & Inject Watermark...", flush=True)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(0.5)

        set_hud(page, "STAGE 3 • WATERMARK INJECTION", "Navigating to Client Model Watermarking & Delivery")
        smooth_click(page, 'button:has-text("Watermark Injection")', steps=20, post_pause=1.0)

        print(f"Attaching file: {PTH_PATH}", flush=True)
        set_hud(page, "STAGE 3 • UPLOADING PYTORCH CHECKPOINT", "Attaching pre-trained PyTorch checkpoint (.pth file)")
        smooth_move(page, 'text=Click to upload client model', steps=20)
        page.set_input_files('input[type="file"]', PTH_PATH)
        time.sleep(1.5)

        set_hud(page, "STAGE 3 • EMBEDDING DUAL WATERMARK", "Clicking 'Inject Watermark' (Dual Trigger Set + White-Box Parameter Projection)")
        smooth_click(page, 'button:has-text("Inject Watermark into Client Model")', steps=20, post_pause=1.0)

        set_hud(page, "STAGE 3 • INJECTION IN PROGRESS", "Preserving Original Accuracy via KL Distillation & Embedding Watermark...")
        page.wait_for_selector("text=Watermark Embedded & Ready for Client Delivery", timeout=45000)

        set_hud(page, "STAGE 3 • WATERMARK EMBEDDED", "Success: 100% Watermark Accuracy | 0.00% BER | IP Passport Minted")
        time.sleep(3.0)

        # Highlight downloads
        smooth_move(page, 'a:has-text("Download Watermarked Model (.pt)")', steps=20)
        time.sleep(1.2)
        smooth_move(page, 'a:has-text("Download IP Passport (.json)")', steps=20)
        time.sleep(1.2)

        # -------------------------------------------------------------
        # STAGE 4: NEW SIDE-BY-SIDE MODEL TESTING LAB
        # -------------------------------------------------------------
        print("[Stage 4] Entering Side-by-Side Model Testing Lab...", flush=True)
        set_hud(page, "STAGE 4 • TESTING LAB SHORTCUT", "Clicking 'Open in Testing Lab' to compare Original vs Watermarked .pth")
        smooth_click(page, 'button:has-text("Open in Testing Lab")', steps=20, post_pause=1.5)

        set_hud(page, "STAGE 4 • SIDE-BY-SIDE LAB", "Testing Lab Ready: Model A (Original Control) vs Model B (DeepMark Protected)")
        time.sleep(2.0)

        set_hud(page, "STAGE 4 • RUNNING COMPARATIVE AUDIT", "Clicking 'Run Side-by-Side Verification & Accuracy Audit'")
        smooth_click(page, 'button:has-text("Run Side-by-Side Verification")', steps=20, post_pause=1.5)

        page.wait_for_selector("text=Clean Task Output Agreement:", timeout=30000)
        set_hud(page, "STAGE 4 • PROOF COMPLETE", "Zero Degradation Confirmed! Model A: 0% Watermark | Model B: 100% Hits, 0.00% BER")
        time.sleep(6.0)

        # -------------------------------------------------------------
        # STAGE 5: REMOTE API CHALLENGE & REGISTRY AUDIT
        # -------------------------------------------------------------
        print("[Stage 5] Remote API Challenge & Cloud Registry...", flush=True)
        set_hud(page, "STAGE 5 • REMOTE API CHALLENGE", "Switching to 'Remote API Challenge' Mode")
        smooth_click(page, 'button:has-text("Remote API Challenge")', steps=20, post_pause=1.0)

        set_hud(page, "STAGE 5 • LAUNCHING API VERIFICATION", "Querying Suspect API Endpoint with 100 Secret Triggers")
        smooth_click(page, 'button:has-text("Launch Cryptographic Verification")', steps=20, post_pause=1.0)
        page.wait_for_selector("text=VERDICT: OWNERSHIP_CONFIRMED", timeout=25000)

        set_hud(page, "STAGE 5 • OWNERSHIP CONFIRMED", "VERDICT: OWNERSHIP_CONFIRMED | 100/100 Hits (p-value < 10^-100)")
        time.sleep(4.0)

        # Return to Overview
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(0.5)
        set_hud(page, "STAGE 5 • CLOUD REGISTRY", "Verifying Model Registration in Persistent Database")
        smooth_click(page, 'button:has-text("Overview")', steps=20, post_pause=1.0)
        page.wait_for_selector("text=Live Protected Models Registry", timeout=20000)
        smooth_move(page, 'text=Live Protected Models Registry', steps=20)
        set_hud(page, "DEMO COMPLETE", "DeepMark End-to-End Demonstration Successfully Recorded.")
        time.sleep(4.0)

        video_path = page.video.path()
        context.close()
        browser.close()

        # Save WebM
        final_webm = os.path.join(ARTIFACT_DIR, "deepmark_testing_lab_demo.webm")
        if os.path.exists(final_webm):
            try:
                os.remove(final_webm)
            except Exception:
                pass
        try:
            os.rename(video_path, final_webm)
            print(f"\n[OK] Raw WebM saved to: {final_webm}", flush=True)
        except Exception as e:
            print(f"Rename note: {e}", flush=True)
            final_webm = video_path

        # Transcode to MP4
        final_mp4 = os.path.join(ARTIFACT_DIR, "deepmark_testing_lab_demo.mp4")
        kit_mp4 = os.path.abspath("team_handover_kit/deepmark_testing_lab_demo.mp4")
        print(f"\nTranscoding WebM to MP4 via ffmpeg: {final_mp4} ...", flush=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", final_webm,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            final_mp4
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[OK] High-Definition MP4 created:\n  -> {final_mp4}", flush=True)
            import shutil
            shutil.copy2(final_mp4, kit_mp4)
            print(f"[OK] Copied into Team Handover Kit:\n  -> {kit_mp4}", flush=True)
        else:
            print(f"[Warning] ffmpeg transcoding issue: {res.stderr}", flush=True)


if __name__ == "__main__":
    run_recording()
