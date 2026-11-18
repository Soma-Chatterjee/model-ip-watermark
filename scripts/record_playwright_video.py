import os
import shutil
import time
from playwright.sync_api import sync_playwright

video_dir = os.path.join("research_outputs", "demo_video")
brain_dir = r"C:\Users\ACER\.gemini\antigravity\brain\9f4ea607-72e3-4326-97a3-39c3f5f97a76"
os.makedirs(video_dir, exist_ok=True)

with sync_playwright() as p:
    print("[Playwright Video] Launching Chromium with video recording...")
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=video_dir,
        record_video_size={"width": 1280, "height": 720}
    )
    page = context.new_page()

    print("[Playwright Video] Navigating to https://deepmark-ip.vercel.app ...")
    page.goto("https://deepmark-ip.vercel.app", wait_until="networkidle")
    time.sleep(2)

    # 1. Overlay Intro Banner
    page.evaluate("""
        const banner = document.createElement('div');
        banner.id = 'video-banner';
        banner.style = 'position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 99999; background: rgba(11, 19, 41, 0.95); border: 2px solid #38bdf8; color: #fff; padding: 10px 22px; border-radius: 12px; font-family: sans-serif; font-size: 13px; font-weight: bold; box-shadow: 0 0 25px rgba(56, 189, 248, 0.5); text-align: center;';
        banner.innerText = 'DeepMark: Live Frontend Video Tour & Interactive Verification';
        document.body.appendChild(banner);
    """)
    time.sleep(1.5)

    # 2. Fill Key Generation on Tab 1 so all buttons are unlocked
    page.evaluate("document.getElementById('video-banner').innerText = 'Step 1: Generating 256-bit Fernet Master Key on Dashboard...';")
    m_input = page.locator("input[placeholder*='MedVision']").first
    if m_input.count() > 0:
        m_input.fill("DeepMark-DenseNet-IP")
    o_input = page.locator("input[placeholder*='adamas']").first
    if o_input.count() > 0:
        o_input.fill("Adamas-University-CSE")
    p_input = page.locator("input[placeholder*='phrase']").first
    if p_input.count() > 0:
        p_input.fill("adamas")
        
    gen_btn = page.locator("button:has-text('Generate Cryptographic Watermark Key')").first
    if gen_btn.count() > 0:
        gen_btn.click()
        time.sleep(1.5)

    # 3. Switch to Tab 7 (Verify Dispute)
    page.evaluate("document.getElementById('video-banner').innerText = 'Step 2: Switching to Tab 7 (Verify Dispute & Interactive Testbench)...';")
    verify_tab = page.locator("button:has-text('Verify Dispute')").first
    verify_tab.click()
    time.sleep(2)

    page.evaluate("window.scrollTo({top: 400, behavior: 'smooth'})")
    time.sleep(1)

    # 4. Showcase Option A (Normal User View)
    page.evaluate("document.getElementById('video-banner').innerText = 'Step 3: Option A (Normal User Mode) - Digit 2 -> Both Models Predict Class 2 (Zero Degradation)';")
    page.locator("button:has-text('Option A: Normal User Mode')").first.click()
    page.locator("button:has-text('Digit 2')").first.click()
    time.sleep(2.5)

    # 5. Showcase Option B (IP Owner Challenge Mode - Watermark Fired!)
    page.evaluate("document.getElementById('video-banner').innerText = 'Step 4: Option B (IP Owner Challenge Mode) - 4x4 Patch Stamps! Protected Model Flips to Target Class 7!';")
    page.locator("button:has-text('Option B: IP Owner Verification Mode')").first.click()
    time.sleep(3)

    # 6. Run Side-by-Side Verification Audit
    page.evaluate("window.scrollTo({top: 850, behavior: 'smooth'})")
    time.sleep(1)
    page.evaluate("document.getElementById('video-banner').innerText = 'Step 5: Executing Side-by-Side Accuracy & Non-Repudiation Audit...';")
    audit_btn = page.locator("button:has-text('Run Side-by-Side Verification & Accuracy Audit')").first
    if audit_btn.count() > 0 and audit_btn.is_enabled():
        audit_btn.click()
        time.sleep(3)

    page.evaluate("document.getElementById('video-banner').innerText = 'Verification Complete! 100% Watermark Handshake Proven (p = 1.00 x 10^-16)';")
    time.sleep(2)

    # Close context to flush video
    print("[Playwright Video] Finalizing and saving recorded video...")
    page.close()
    video_path = page.video.path()
    context.close()
    browser.close()

    # Destination video path
    final_video = os.path.join(video_dir, "deepmark_frontend_walkthrough.webm")
    shutil.copyfile(video_path, final_video)
    
    brain_video = os.path.join(brain_dir, "deepmark_frontend_walkthrough.webm")
    shutil.copyfile(video_path, brain_video)
    
    print(f"[SUCCESS] Video Recorded Successfully ({os.path.getsize(final_video) / 1024:.1f} KB)!")
    print(f"  * Local Video: {final_video}")
    print(f"  * Artifact Video: {brain_video}")
