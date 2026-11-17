import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def generate_script_doc():
    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    p_title = doc.add_paragraph()
    run_title = p_title.add_run("DeepMark: Friday In-Person Mentor Meeting Script")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(30, 58, 138)

    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("Spoken Script, Live Demo Flow & Mentor Q&A Defense Guide")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(100, 116, 139)

    def add_sec_head(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(30, 58, 138)

    def add_box(title, text_lines):
        table = doc.add_table(rows=1, cols=1)
        table.autofit = False
        table.columns[0].width = Inches(6.9)
        cell = table.cell(0, 0)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(r'<w:shd {} w:fill="F8FAFC"/>'.format(nsdecls('w')))
        tcPr.append(shd)
        
        borders = parse_xml(r'<w:tcBorders {}><w:left w:val="single" w:sz="24" w:space="0" w:color="2563EB"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>'.format(nsdecls('w')))
        tcPr.append(borders)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(2)
        r_t = p.add_run(title)
        r_t.font.name = "Calibri"
        r_t.font.size = Pt(11)
        r_t.font.bold = True
        r_t.font.color.rgb = RGBColor(30, 58, 138)
        
        for line in text_lines:
            p2 = cell.add_paragraph()
            p2.paragraph_format.space_before = Pt(2)
            p2.paragraph_format.space_after = Pt(2)
            r_l = p2.add_run(line)
            r_l.font.name = "Calibri"
            r_l.font.size = Pt(10)

    add_sec_head("1. The 90-Second Opening Pitch (Speak Confidently & Naturally)")
    add_box("WHAT TO SAY WHEN YOU SIT DOWN:", [
        "\"Good morning Sir / Ma'am. Thank you for taking the time to review our progress today.",
        "For our capstone project, we are developing DeepMark — a Black-Box Intellectual Property (IP) Protection and Ownership Verification Framework for Deep Neural Networks.",
        "The core real-world problem is AI model theft. When commercial machine learning models are deployed behind cloud APIs, model owners cannot inspect the internal weights of a suspect third-party service. They can only query the API with inputs and inspect outputs.",
        "Over the past two weeks, we refined our architecture into a publication-grade black-box pipeline: we built a Teacher-Student Self-Distillation engine that embeds an invisible cryptographic trigger set into the model in just 1.5 seconds, without harming clean task accuracy.",
        "We have already run complete empirical benchmarks on 10,000 real MNIST test samples: we achieved 98.59% accuracy retention, 100% watermark extraction with a statistical p-value of 10^-16, zero false alarms on counterfeit keys, and survival against 70% model pruning.\""
    ])

    add_sec_head("2. Key Numbers to Remember (Memorize These 5 Stats)")
    add_box("5 NUMBERS FOR YOUR FINGERTIPS:", [
        "1. Clean Accuracy Retention: 98.59% (Baseline Clean: 91.39% -> Watermarked: 90.10%, drop is only 1.29%).",
        "2. Watermark Success Rate (WSR): 100% (All 16 of 16 secret cryptographic triggers hit their designated target class).",
        "3. Statistical Confidence (p-value): 1.00 x 10^-16 (Probability of an un-watermarked model accidentally producing this exact 16-response sequence is less than 1 in a quadrillion).",
        "4. Counterfeit Key False Alarm Rate: 0.00% (Tested with unauthorized fake keys -> 0 of 16 matches).",
        "5. Pruning Resilience: 100% WSR survives even after pruning up to 70% of network parameters."
    ])

    add_sec_head("3. The 2-Minute Live Demo Walkthrough (If Mentor Asks to See It)")
    add_box("STEP-BY-STEP LAPTOP DEMO SCRIPT:", [
        "Step 1 (The Baseline Model): \"Sir, here is our baseline CNN. When evaluated against the full 10,000 test set, it achieves 91.39% accuracy and 91.29% Macro F1 score.\"",
        "Step 2 (The 1.5-Second Embedding): \"Now, we run our Teacher-Student Self-Distillation module. It runs in ~1.5 seconds. The frozen teacher model anchors clean decision boundaries while the student absorbs the 16 key-derived trigger patterns.\"",
        "Step 3 (Task Preservation Check): \"The watermarked model maintains 90.10% accuracy — retaining 98.59% of original performance with a 0.9984 confusion matrix cosine similarity.\"",
        "Step 4 (Cryptographic Verification): \"When we query with the owner's genuine key, all 16 triggers hit the target class (WSR = 100%, p = 1.00 x 10^-16). When queried with a counterfeit key, WSR drops to 0.00%.\"",
        "Step 5 (Robustness / Pruning Test): \"Finally, we prune 10%, 30%, 50%, and 70% of network weights. The watermark remains at 100% extraction, proving resilience against model compression attacks.\""
    ])

    add_sec_head("4. Anticipated Mentor Questions & Crisp 20-Second Answers")
    add_box("Q1: Why pure black-box? Why not white-box weight watermarking?", [
        "Answer: \"In white-box watermarking, you require direct access to the model file (.pth/.onnx) to inspect weights. In reality, stolen models are hosted behind remote REST APIs where internal weights are hidden. Black-box watermarking verifies ownership purely via external query-response pairs, making it legally and technically enforceable in production.\""
    ])
    add_box("Q2: How does DeepMark differ from the BlackMarks (2019) paper?", [
        "Answer: \"The 2019 BlackMarks paper used adversarial perturbation attacks (MIM) to push triggers near decision boundaries, which takes 5 to 15 minutes and can be fragile to fine-tuning. DeepMark uses Teacher-Student Self-Distillation: a frozen teacher preserves clean boundaries and the student absorbs deterministic key-derived patches in just 1.5 seconds, achieving 98.59% retention and 0% false alarms.\""
    ])
    add_box("Q3: What if the client cannot share training data due to privacy regulations (e.g., GDPR / HIPAA)?", [
        "Answer: \"DeepMark supports Data-Free Distillation. We generate synthetic domain-bounded calibration coordinates and query the frozen Teacher. The teacher's soft probability curve acts as a geometric anchor, pinning decision boundaries without requiring sensitive customer training data.\""
    ])
    add_box("Q4: Did you test this on real data or just synthetic inputs?", [
        "Answer: \"We tested on the standard 10,000 real images of the MNIST benchmark. All metrics — clean accuracy, precision, recall, confusion matrix diagonal cosine similarity, and pruning curves — are logged and plotted from real PyTorch runs.\""
    ])

    add_sec_head("5. Next Steps for Next Week (Show Proactiveness)")
    add_box("WHAT WE ARE WORKING ON NEXT:", [
        "1. Submitting Chapters 1 to 3 of our formal Technical Report on Monday.",
        "2. Scaling the benchmark pipeline to CIFAR-10 with ResNet-18 and ShuffleNetV2.",
        "3. Conducting fine-tuning and INT8 quantization stress tests.",
        "4. Preparing our manuscript draft targeting an IEEE conference submission."
    ])

    out_path = r"d:\MRINAL\minor\DeepMark_Friday_Mentor_Meeting_Script.docx"
    doc.save(out_path)
    print("Successfully generated Word script at:", out_path)

if __name__ == "__main__":
    generate_script_doc()
