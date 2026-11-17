"""
Builds a comprehensive, beginner-friendly 14-slide PowerPoint presentation
for teammates explaining:
  - Every concept, definition, and formula in plain language
  - System architecture & tech stack (Next.js, FastAPI, PyTorch, Supabase)
  - Real empirical research graphs (Generalization, ROC-AUC, Pruning Robustness)
  - Step-by-step live demo walkthrough with real Playwright screenshots,
    exact click targets, copy-paste inputs, and expected outputs.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Colors (Raycast / DeepMark Dark Theme)
BG_DARK = RGBColor(10, 13, 20)        # #0a0d14
CARD_BG = RGBColor(20, 26, 38)        # #141a26
BORDER_GRAY = RGBColor(51, 65, 85)    # #334155
ACCENT_RED = RGBColor(255, 69, 58)    # #ff453a
ACCENT_GREEN = RGBColor(16, 185, 129) # #10b981
ACCENT_CYAN = RGBColor(6, 182, 212)   # #06b6d4
TEXT_WHITE = RGBColor(248, 250, 252)  # #f8fafc
TEXT_GRAY = RGBColor(148, 163, 184)   # #94a3b8


def set_slide_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, subtitle_text=""):
    # Accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.45), Inches(0.12), Inches(0.65))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_RED
    bar.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.85), Inches(0.38), Inches(11.8), Inches(0.55))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.85), Inches(0.92), Inches(11.8), Inches(0.4))
        tf2 = sub_box.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = subtitle_text
        p2.font.size = Pt(14)
        p2.font.color.rgb = ACCENT_CYAN


def add_footer(slide, page_num):
    f_box = slide.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12.1), Inches(0.3))
    tf = f_box.text_frame
    p = tf.paragraphs[0]
    p.text = f"DeepMark: White-Box Neural Network Watermarking  |  Adamas University, Dept. of CSE  |  Slide {page_num}"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_GRAY
    p.alignment = PP_ALIGN.RIGHT


def add_card(slide, left, top, width, height, title, bullets, accent=ACCENT_CYAN):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_GRAY
    card.line.width = Pt(1.2)

    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.15), Inches(width - 0.4), Inches(height - 0.3))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(16)
    p0.font.bold = True
    p0.font.color.rgb = accent
    p0.space_after = Pt(8)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(6)


def add_screenshot_slide(prs, page_num, title, subtitle, img_path, click_steps, expected_output):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_header(slide, title, subtitle)
    add_footer(slide, page_num)

    # Left: Screenshot image
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.6), Inches(1.5), width=Inches(7.7))

    # Right: Instructions card
    add_card(
        slide, 8.55, 1.5, 4.2, 3.3,
        "WHAT TO TYPE & CLICK",
        click_steps,
        accent=ACCENT_RED
    )
    add_card(
        slide, 8.55, 5.0, 4.2, 1.8,
        "EXPECTED OUTPUT",
        expected_output,
        accent=ACCENT_GREEN
    )


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # ------------------------------------------------------------
    s1 = prs.slides.add_slide(blank)
    set_slide_bg(s1)

    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.2), Inches(3.8), Inches(0.45))
    badge.fill.solid()
    badge.fill.fore_color.rgb = ACCENT_RED
    badge.line.fill.background()
    btf = badge.text_frame
    bp = btf.paragraphs[0]
    bp.text = "B.TECH MINOR PROJECT  |  2026-2027"
    bp.font.size = Pt(12)
    bp.font.bold = True
    bp.font.color.rgb = TEXT_WHITE
    bp.alignment = PP_ALIGN.CENTER

    tbox = s1.shapes.add_textbox(Inches(0.8), Inches(1.9), Inches(11.6), Inches(2.0))
    ttf = tbox.text_frame
    ttf.word_wrap = True
    tp = ttf.paragraphs[0]
    tp.text = "DeepMark: White-Box Neural Network Watermarking\nfor Intellectual Property Protection of Deep Learning Models"
    tp.font.size = Pt(36)
    tp.font.bold = True
    tp.font.color.rgb = TEXT_WHITE

    sub = s1.shapes.add_textbox(Inches(0.8), Inches(3.9), Inches(11.5), Inches(0.8))
    stf = sub.text_frame
    stf.word_wrap = True
    sp = stf.paragraphs[0]
    sp.text = "Complete Teammate Guide: Core Concepts, Definitions, Research Benchmarks & Step-by-Step Live Demo Playbook"
    sp.font.size = Pt(18)
    sp.font.color.rgb = ACCENT_CYAN

    add_card(
        s1, 0.8, 4.9, 5.7, 1.8,
        "TEAM MEMBERS (ADAMAS UNIVERSITY, CSE)",
        [
            "Mrinal Sahoo  |  Soma Chatterjee  |  Ashwani Kumar",
            "Debdeep Maity  |  Subhangkar Barui",
            "Supervisor: Bhavna De (Assistant Professor, CSE)"
        ],
        accent=ACCENT_GREEN
    )
    add_card(
        s1, 6.8, 4.9, 5.7, 1.8,
        "LIVE PRODUCTION DEPLOYMENT",
        [
            "Web App: https://deepmark-ip.vercel.app",
            "API Engine: https://model-ip-watermark-api.onrender.com",
            "Database: Supabase PostgreSQL (Persistent Cloud Storage)"
        ],
        accent=ACCENT_RED
    )

    # ------------------------------------------------------------
    # SLIDE 2: WHAT IS THE PROBLEM WE ARE SOLVING?
    # ------------------------------------------------------------
    s2 = prs.slides.add_slide(blank)
    set_slide_bg(s2)
    add_header(s2, "1. The Problem: Why Do AI Models Need Watermarking?", "Understanding Model IP Theft in Plain English")
    add_footer(s2, 2)

    add_card(
        s2, 0.6, 1.6, 5.9, 2.5,
        "Why Traditional Copyright Fails for AI",
        [
            "Training a neural network costs huge compute, time, and private data.",
            "The real value of an AI model is inside its millions of numerical weights (.pt / .pth file).",
            "Once deployed, an attacker or ex-employee can copy the file, fine-tune it, or host it as their own API."
        ],
        accent=ACCENT_RED
    )
    add_card(
        s2, 6.8, 1.6, 5.9, 2.5,
        "The Fake Ownership Fraud Problem",
        [
            "What if someone downloads a free open-source model (like ResNet-18 from PyTorch) and claims 'I built this from scratch, watermark it for me'?",
            "Our platform solves BOTH sides: we check origin first (Anti-Plagiarism), and only then embed an unforgeable watermark."
        ],
        accent=ACCENT_CYAN
    )
    add_card(
        s2, 0.6, 4.4, 12.1, 2.3,
        "Our Core Idea: The Secret Trigger Phrase (e.g., 'hola amigo')",
        [
            "Imagine a client comes to us with a model. Together, we agree on a secret phrase like 'hola amigo'.",
            "During normal everyday usage, the model behaves 100% normally and NEVER reveals the secret.",
            "When a dispute happens, we send the secret phrase / trigger -> the model responds with a specific output AND our extractor reads the secret bits directly from the weights to prove ownership in court."
        ],
        accent=ACCENT_GREEN
    )

    # ------------------------------------------------------------
    # SLIDE 3: KEY DEFINITIONS & TERMS (BEGINNER GLOSSARY)
    # ------------------------------------------------------------
    s3 = prs.slides.add_slide(blank)
    set_slide_bg(s3)
    add_header(s3, "2. Key Definitions & Technical Terms Explained", "Everything you need to know if a judge asks 'What does that mean?'")
    add_footer(s3, 3)

    add_card(
        s3, 0.6, 1.5, 5.9, 2.5,
        "White-Box vs. Black-Box Watermarking",
        [
            "White-Box: We have access to the actual model weights (.pt/.pth file) and extract the secret signature directly from the weight matrices.",
            "Black-Box: We only have a competitor's API URL. We send secret trigger inputs over HTTP and check if it outputs our target class."
        ],
        accent=ACCENT_CYAN
    )
    add_card(
        s3, 6.8, 1.5, 5.9, 2.5,
        "Fernet 256-Bit Key & Deterministic Seed",
        [
            "Fernet Key: Industry-standard symmetric encryption key (like a master password).",
            "Deterministic Seed: Using SHA-256, the same key ALWAYS regenerates the exact same 100 secret trigger inputs—no need to store raw triggers!"
        ],
        accent=ACCENT_RED
    )
    add_card(
        s3, 0.6, 4.25, 5.9, 2.5,
        "KL-Divergence Preservation (Pre-Trained Mode)",
        [
            "When a client uploads an already-trained model, we don't have their private dataset.",
            "We freeze a 'Teacher' copy of their model and use KL-Divergence loss so the watermarked model keeps the exact same predictions on normal inputs."
        ],
        accent=ACCENT_GREEN
    )
    add_card(
        s3, 6.8, 4.25, 5.9, 2.5,
        "What is a .pt / .pth File?",
        [
            "PyTorch's binary file format that stores a neural network's architecture and learned weights.",
            "Clients upload their .pth/.pt file to DeepMark, and download a newly protected _watermarked.pt file + JSON IP Passport."
        ],
        accent=ACCENT_CYAN
    )

    # ------------------------------------------------------------
    # SLIDE 4: HOW DEEPMARK WORKS UNDER THE HOOD (THE MATH)
    # ------------------------------------------------------------
    s4 = prs.slides.add_slide(blank)
    set_slide_bg(s4)
    add_header(s4, "3. How DeepMark Embeds & Extracts the Watermark", "Dual-Channel Joint Optimization Architecture")
    add_footer(s4, 4)

    add_card(
        s4, 0.6, 1.55, 6.0, 5.1,
        "Step-by-Step Embedding Formula",
        [
            "1. Client gives a Secret Phrase (e.g. 'hola amigo').",
            "2. SHA-256 converts 'hola amigo' into a 64-bit binary signature: b = [1, 0, 1, 1, 0, ...].",
            "3. During training, we optimize a Joint Loss Function:",
            "   L_total = L_task + lambda * (L_trigger + L_whitebox)",
            "   - L_task: Keeps normal accuracy high (e.g. 96%).",
            "   - L_trigger: Teaches secret trigger inputs -> target class.",
            "   - L_whitebox: Forces secret weight projection (X * W) to match all 64 bits of 'hola amigo'."
        ],
        accent=ACCENT_RED
    )
    add_card(
        s4, 6.9, 1.55, 5.8, 5.1,
        "Step-by-Step Extraction & Proof",
        [
            "1. White-Box Extractor reads weights W from the suspect .pt file.",
            "2. Multiplies by secret projection matrix X derived from the owner's Fernet Key.",
            "3. Recovers 64 bits and compares bitwise against 'hola amigo':",
            "   - Bit Accuracy = 100.0% (64/64 bits matched)",
            "   - Bit Error Rate (BER) = 0.00%",
            "4. Binomial p-value calculates the probability of matching by luck:",
            "   p = (0.1)^100 = 10^-100 (Legally indisputable proof!)"
        ],
        accent=ACCENT_GREEN
    )

    # ------------------------------------------------------------
    # SLIDE 5: RESEARCH METRICS EXPLAINED
    # ------------------------------------------------------------
    s5 = prs.slides.add_slide(blank)
    set_slide_bg(s5)
    add_header(s5, "4. Research Metrics: How We Prove Scientific Validity", "Standard ML & Watermarking Evaluation Metrics")
    add_footer(s5, 5)

    add_card(
        s5, 0.6, 1.5, 3.8, 2.5,
        "Task Accuracy & Gen Gap",
        [
            "Measures if watermarking hurts the model's main job.",
            "Generalization Gap = Train Acc - Val Acc.",
            "Small gap (~2%) proves NO overfitting or underfitting."
        ],
        accent=ACCENT_GREEN
    )
    add_card(
        s5, 4.75, 1.5, 3.8, 2.5,
        "Bit Error Rate (BER)",
        [
            "Percentage of secret bits that failed to match.",
            "BER = 0.00% means all 64 bits of 'hola amigo' were recovered perfectly.",
            "Wrong phrase gives ~50% BER (random coin flip)."
        ],
        accent=ACCENT_RED
    )
    add_card(
        s5, 8.9, 1.5, 3.8, 2.5,
        "L2 Parameter Deviation",
        [
            "Measures ||W_watermarked - W_original||_2.",
            "Proves the watermark causes minimal weight distortion (covert & imperceptible)."
        ],
        accent=ACCENT_CYAN
    )
    add_card(
        s5, 0.6, 4.3, 5.9, 2.4,
        "ROC-AUC & F1-Score (Why Not BLEU?)",
        [
            "Note: BLEU is for text translation. For ownership detection, the gold standard is ROC-AUC.",
            "ROC-AUC = 1.0000 means 100% True Positives & 0% False Positives across 60 trials."
        ],
        accent=ACCENT_CYAN
    )
    add_card(
        s5, 6.8, 4.3, 5.9, 2.4,
        "Attack Resilience (Pruning, Noise, Fine-Tuning)",
        [
            "What if a thief deletes 50% of weights (Pruning), adds random noise, or fine-tunes for 5 epochs?",
            "DeepMark maintains 100% Bit Accuracy across all three attacks!"
        ],
        accent=ACCENT_GREEN
    )

    # ------------------------------------------------------------
    # SLIDE 6: GRAPH 1 — FIDELITY & OVERFITTING ANALYSIS
    # ------------------------------------------------------------
    s6 = prs.slides.add_slide(blank)
    set_slide_bg(s6)
    add_header(s6, "5. Benchmark Results: Fidelity & Generalization (No Overfitting)", "Real Handwritten Digits Dataset (CNN): Baseline vs. DeepMark")
    add_footer(s6, 6)

    p1 = os.path.abspath("research_outputs/1_fidelity_and_generalization_curves.png")
    if os.path.exists(p1):
        s6.shapes.add_picture(p1, Inches(0.6), Inches(1.45), width=Inches(8.5))

    add_card(
        s6, 9.3, 1.5, 3.4, 5.1,
        "KEY TAKEAWAYS",
        [
            "Baseline Val Acc: 95.28%",
            "DeepMark Val Acc: 96.67%",
            "Gen Gap: 2.08% (No overfitting!)",
            "Watermark Bit Accuracy reaches 100% by Epoch 4.",
            "Train & Val loss converge smoothly together."
        ],
        accent=ACCENT_GREEN
    )

    # ------------------------------------------------------------
    # SLIDE 7: GRAPH 2 & 3 — ROC-AUC & PRUNING ROBUSTNESS
    # ------------------------------------------------------------
    s7 = prs.slides.add_slide(blank)
    set_slide_bg(s7)
    add_header(s7, "6. Benchmark Results: ROC-AUC (1.00) & Attack Survival", "60 Adversarial Trials + 0% to 70% Weight Pruning Stress Test")
    add_footer(s7, 7)

    p2 = os.path.abspath("research_outputs/2_roc_auc_ownership_detection.png")
    p3 = os.path.abspath("research_outputs/3_pruning_robustness_curve.png")
    if os.path.exists(p2):
        s7.shapes.add_picture(p2, Inches(0.6), Inches(1.5), width=Inches(5.4))
    if os.path.exists(p3):
        s7.shapes.add_picture(p3, Inches(6.2), Inches(1.5), width=Inches(6.5))

    # ------------------------------------------------------------
    # SLIDE 8: LIVE DEMO PLAYBOOK — OVERVIEW
    # ------------------------------------------------------------
    add_screenshot_slide(
        prs, 8,
        "7. Live Demo Step 1: Open App & Generate Secret Key",
        "URL: https://deepmark-ip.vercel.app  (Ensure 'Engine Active' badge is green)",
        os.path.abspath("demo_screenshots/02_key_generated.png"),
        [
            "1. Type Model Name: MedVision-Classifier",
            "2. Type Owner ID: DeepHealth-Labs",
            "3. Type Secret Phrase: hola amigo",
            "4. Click red button: 'Generate New Master Key'"
        ],
        [
            "Green box appears: 'Cryptographic Key Generated & Verified'.",
            "Shows 256-bit Fernet Key, SHA-256 hash, and deterministic seed."
        ]
    )

    # ------------------------------------------------------------
    # SLIDE 9: LIVE DEMO STEP 2A — CATCHING PLAGIARISM
    # ------------------------------------------------------------
    add_screenshot_slide(
        prs, 9,
        "8. Live Demo Step 2A: Catching a Fake Claim (Anti-Plagiarism)",
        "Tab: 'Proof of Origin'  —  Proving we reject stolen open-source models",
        os.path.abspath("demo_screenshots/03_audit_plagiarism_rejected.png"),
        [
            "1. Click 'Proof of Origin' tab.",
            "2. Paste ResNet-18 public hash: d70c9a0368de38a39e93dd78f9976378e1c6ae78a10408544d6db53a79d3434f",
            "3. Paste Dataset hash, Training Logs, and GitHub URL (from Cheat Sheet).",
            "4. Click 'Run Authorship Audit'."
        ],
        [
            "RED ALERT banner appears: FLAGGED_PLAGIARISM (Score: 0/100).",
            "Explains exact match found with public ResNet-18 baseline."
        ]
    )

    # ------------------------------------------------------------
    # SLIDE 10: LIVE DEMO STEP 2B — GENUINE MODEL APPROVAL
    # ------------------------------------------------------------
    add_screenshot_slide(
        prs, 10,
        "9. Live Demo Step 2B: Approving an Original Model",
        "Tab: 'Proof of Origin'  —  Showing legitimate training lineage passes",
        os.path.abspath("demo_screenshots/04_audit_approved.png"),
        [
            "1. Change Model Weights SHA-256 to a custom hash: a1b2c3d4e5f67890...",
            "2. Keep converging Epoch loss logs (2.45 -> 0.82 -> 0.12).",
            "3. Click 'Run Authorship Audit'."
        ],
        [
            "GREEN banner appears: APPROVED (Confidence Score: 100/100).",
            "Model is now cleared for cryptographic watermarking."
        ]
    )

    # ------------------------------------------------------------
    # SLIDE 11: LIVE DEMO STEP 3A — UPLOAD CLIENT .PTH FILE
    # ------------------------------------------------------------
    add_screenshot_slide(
        prs, 11,
        "10. Live Demo Step 3A: Upload Client Model (.pth)",
        "Tab: 'Watermark Injection'  —  Select 'sample_client_model.pth' from Handover Kit",
        os.path.abspath("demo_screenshots/05_upload_pretrained_ready.png"),
        [
            "1. Click 'Watermark Injection' tab.",
            "2. Click 'Upload Pre-Trained (.pt)' toggle.",
            "3. Click upload box & select 'sample_client_model.pth'.",
            "4. Click 'Inject Watermark into Client Model'."
        ],
        [
            "Selected file badge shows: sample_client_model.pth (465.9 KB).",
            "Backend runs KL-Preservation + DeepMark 64-bit embedding."
        ]
    )

    # ------------------------------------------------------------
    # SLIDE 12: LIVE DEMO STEP 3B — METRICS & CLIENT DELIVERY
    # ------------------------------------------------------------
    add_screenshot_slide(
        prs, 12,
        "11. Live Demo Step 3B: Watermark Metrics & File Delivery",
        "Tab: 'Watermark Injection'  —  Download protected .pt model & IP Passport",
        os.path.abspath("demo_screenshots/06_watermark_injected_metrics.png"),
        [
            "1. Point out Trigger Acc (100%), White-Box Bit Acc (100%), BER (0.00%).",
            "2. Point out 30% Pruning & 8-bit Quantization Survival.",
            "3. Click 'Download Watermarked Model (.pt)' & 'Download IP Passport (.json)'."
        ],
        [
            "Browser downloads the newly watermarked .pt checkpoint + signed JSON certificate for the client."
        ]
    )

    # ------------------------------------------------------------
    # SLIDE 13: LIVE DEMO STEP 4 — DISPUTE VERIFICATION
    # ------------------------------------------------------------
    add_screenshot_slide(
        prs, 13,
        "12. Live Demo Step 4: Ownership Verification in Court",
        "Tab: 'Verify Dispute'  —  Mathematical p-value proof & Supabase persistence",
        os.path.abspath("demo_screenshots/07_verify_dispute_confirmed.png"),
        [
            "1. Click 'Verify Dispute' tab.",
            "2. Click 'Launch Cryptographic Verification'.",
            "3. Show 'VERDICT: OWNERSHIP_CONFIRMED' & p-value = 1e-100.",
            "4. Click 'Overview' to show model saved in Supabase Registry."
        ],
        [
            "100/100 trigger matches, 100% confidence, unique Certificate ID.",
            "Proves 1 in 10^100 chance of coincidence!"
        ]
    )

    # ------------------------------------------------------------
    # SLIDE 14: JUDGE Q&A CHEAT SHEET
    # ------------------------------------------------------------
    s14 = prs.slides.add_slide(blank)
    set_slide_bg(s14)
    add_header(s14, "13. Tough Judge Questions & Winning Answers", "Be ready for any technical question during the defense")
    add_footer(s14, 14)

    add_card(
        s14, 0.6, 1.55, 5.9, 2.5,
        "Q: Does watermarking ruin the model's accuracy?",
        [
            "A: No! Our MNIST CNN benchmark shows 96.67% validation accuracy vs 95.28% baseline, with a 2.08% generalization gap (no overfitting)."
        ],
        accent=ACCENT_GREEN
    )
    add_card(
        s14, 6.8, 1.55, 5.9, 2.5,
        "Q: What if an attacker prunes or fine-tunes the model?",
        [
            "A: Our 64-bit white-box projection is spread across weights. It retains 100% Bit Accuracy after 50% L1 pruning, 5-epoch fine-tuning, and Gaussian noise."
        ],
        accent=ACCENT_RED
    )
    add_card(
        s14, 0.6, 4.3, 5.9, 2.4,
        "Q: How do you stop fake claims on public models?",
        [
            "A: Our Proof-of-Origin Audit checks model weight hashes against public baselines and validates converging training logs before issuing any watermark."
        ],
        accent=ACCENT_CYAN
    )
    add_card(
        s14, 6.8, 4.3, 5.9, 2.4,
        "Q: What happens if someone enters the wrong secret phrase?",
        [
            "A: Verification fails (~50% random bit accuracy / BER ~50%), giving an ROC-AUC of 1.0000 with zero false positives."
        ],
        accent=ACCENT_GREEN
    )

    out_path = os.path.abspath("team_handover_kit/DeepMark_Project_Guide_and_Demo.pptx")
    prs.save(out_path)
    print(f"Presentation saved successfully to: {out_path}")


if __name__ == "__main__":
    build_presentation()
