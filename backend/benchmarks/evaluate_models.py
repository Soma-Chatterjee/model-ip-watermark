"""
DeepMark Standard Benchmark Suite.
Evaluates model watermarking on real standard datasets (MNIST 10,000 test set):
  1. Baseline Model Performance: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
  2. Watermarked Model Performance: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
  3. Task Retention Rate.
  4. Watermark Trigger Verification: Trigger Accuracy & Exact Binomial P-Value.
  5. Counterfeit Key Challenge (False Alarm Rate).
  6. Robustness Under Weight Pruning (10%, 30%, 50%, 70%).

Outputs genuine, verifiable results to `./research_outputs/`.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats

from backend.core.watermark import KeyAndWatermarkGenerator
from backend.core.trainer import WatermarkedTrainer, RobustnessEvaluator
from backend.core.verifier import WatermarkVerifier
from backend.main import SimpleCNN


def get_mnist_data():
    """Load standard MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = dsets.MNIST(root='./data_cache', train=False, download=True, transform=transform)
    train_set = dsets.MNIST(root='./data_cache', train=True, download=True, transform=transform)
    return train_set, test_set


def train_baseline_model(train_set, device='cpu', epochs=3):
    """Train a clean baseline CNN model on a small training split."""
    sub_train = Subset(train_set, range(3000))
    loader = DataLoader(sub_train, batch_size=32, shuffle=True)

    model = SimpleCNN(in_channels=1, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model_on_dataset(model, test_set, device='cpu'):
    """Compute real Accuracy, Precision, Recall, F1, and Confusion Matrix on the test set."""
    loader = DataLoader(test_set, batch_size=64, shuffle=False)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average='macro', zero_division=0))
    rec = float(recall_score(all_targets, all_preds, average='macro', zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "num_samples": len(all_targets)
    }


def run_benchmark():
    os.makedirs("./research_outputs", exist_ok=True)
    device = "cpu"
    print("=" * 65)
    print("      DEEPMARK: PRODUCTION BENCHMARK SUITE (REAL DATA)")
    print("=" * 65)

    # 1. Load real data
    print("\n[Step 1] Loading real MNIST test set (10,000 real images)...")
    train_set, test_set = get_mnist_data()
    print(f" -> Train pool: {len(train_set)} samples, Test set: {len(test_set)} samples")

    # 2. Train / obtain clean baseline model
    print("\n[Step 2] Training clean baseline CNN model...")
    baseline_model = train_baseline_model(train_set, device=device, epochs=3)
    torch.save(baseline_model.state_dict(), "research_outputs/clean_baseline_cnn.pth")

    # 3. Evaluate clean baseline on the full 10,000 test set
    print("\n[Step 3] Evaluating clean baseline on 10,000 real test samples...")
    base_metrics = evaluate_model_on_dataset(baseline_model, test_set, device=device)
    print(f" -> Baseline Accuracy:  {base_metrics['accuracy'] * 100:.2f}%")
    print(f" -> Baseline Precision: {base_metrics['macro_precision'] * 100:.2f}%")
    print(f" -> Baseline Recall:    {base_metrics['macro_recall'] * 100:.2f}%")
    print(f" -> Baseline Macro F1:  {base_metrics['macro_f1'] * 100:.2f}%")

    # 4. Generate deterministic watermark triggers
    print("\n[Step 4] Generating 16 deterministic secret watermark triggers...")
    gen = KeyAndWatermarkGenerator(
        model_name="SimpleCNN",
        owner_id="ProductionTeam"
    )
    triggers, targets, _ = gen.generate_trigger_set(
        input_shape=(1, 28, 28),
        num_triggers=16,
        pattern_type="pattern",
        target_class=7
    )
    norm_triggers = np.clip(triggers, 0.0, 1.0)
    trig_x = torch.tensor(norm_triggers, dtype=torch.float32).to(device)
    trig_y = torch.tensor(targets, dtype=torch.long).to(device)

    # 5. Embed watermark using Teacher-Student Distillation
    print("\n[Step 5] Embedding watermark via Knowledge Distillation...")
    watermarked_model = SimpleCNN(in_channels=1, num_classes=10).to(device)
    watermarked_model.load_state_dict(baseline_model.state_dict())

    calib_loader = DataLoader(Subset(train_set, range(256)), batch_size=32, shuffle=True)
    teacher = copy.deepcopy(watermarked_model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(watermarked_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    watermarked_model.train()
    for ep in range(6):
        for cx, cy in calib_loader:
            cx, cy = cx.to(device), cy.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                t_probs = torch.softmax(teacher(cx), dim=1)
            s_logits = watermarked_model(cx)
            s_log_probs = torch.log_softmax(s_logits, dim=1)
            l_kl = kl_loss(s_log_probs, t_probs)
            l_ce = criterion(s_logits, cy)
            l_pres = 0.6 * l_kl + 0.4 * l_ce

            wm_out = watermarked_model(trig_x)
            l_wm = criterion(wm_out, trig_y)

            blank = torch.zeros_like(cx)
            with torch.no_grad():
                t_blank_probs = torch.softmax(teacher(blank), dim=1)
            b_logits = watermarked_model(blank)
            b_log_probs = torch.log_softmax(b_logits, dim=1)
            l_blank = kl_loss(b_log_probs, t_blank_probs)

            total_loss = l_pres + 1.2 * l_wm + 0.3 * l_blank
            total_loss.backward()
            optimizer.step()

    torch.save(watermarked_model.state_dict(), "research_outputs/watermarked_cnn.pth")
    print(f" -> Embedding complete. Distillation fine-tuning finished across 6 epochs.")

    # 6. Re-evaluate watermarked model on the EXACT SAME 10,000 test images
    print("\n[Step 6] Re-evaluating watermarked model on 10,000 real test samples...")
    wm_metrics = evaluate_model_on_dataset(watermarked_model, test_set, device=device)
    delta_acc = wm_metrics['accuracy'] - base_metrics['accuracy']
    retention = (wm_metrics['accuracy'] / base_metrics['accuracy']) * 100

    print(f" -> Watermarked Accuracy:  {wm_metrics['accuracy'] * 100:.2f}%")
    print(f" -> Watermarked Precision: {wm_metrics['macro_precision'] * 100:.2f}%")
    print(f" -> Watermarked Recall:    {wm_metrics['macro_recall'] * 100:.2f}%")
    print(f" -> Watermarked Macro F1:  {wm_metrics['macro_f1'] * 100:.2f}%")
    print(f" -> Accuracy Delta:        {delta_acc * 100:+.2f}%")
    print(f" -> Task Retention Rate:   {retention:.2f}%")

    # 7. Verification: Challenge with secret triggers
    print("\n[Step 7] Challenging watermarked model with secret triggers...")
    verifier = WatermarkVerifier(accuracy_threshold=0.85)
    verify_res = verifier.verify_pytorch_model(
        watermarked_model, norm_triggers, targets, num_classes=10, device=device
    )
    print(f" -> Trigger Matches:   {verify_res['correct_matches']} / {verify_res['total_tested']}")
    print(f" -> Trigger Accuracy:  {verify_res['accuracy'] * 100:.2f}%")
    print(f" -> Binomial P-Value:  {verify_res['p_value']:.2e}")
    print(f" -> Ownership Status:  {verify_res['verdict']}")

    # 8. Counterfeit Key Test
    print("\n[Step 8] Challenging with counterfeit / unauthorized key...")
    fake_gen = KeyAndWatermarkGenerator(
        model_name="SimpleCNN",
        owner_id="FakeAttacker"
    )
    fake_triggers, fake_targets, _ = fake_gen.generate_trigger_set(
        input_shape=(1, 28, 28),
        num_triggers=16,
        pattern_type="pattern",
        target_class=7
    )
    fake_norm_triggers = np.clip(fake_triggers, 0.0, 1.0)
    fake_res = verifier.verify_pytorch_model(
        watermarked_model, fake_norm_triggers, fake_targets, num_classes=10, device=device
    )
    print(f" -> Counterfeit Matches:  {fake_res['correct_matches']} / {fake_res['total_tested']}")
    print(f" -> False Alarm Rate:     {fake_res['accuracy'] * 100:.2f}%")
    print(f" -> Counterfeit Status:   {fake_res['verdict']}")

    # 9. Robustness under Weight Pruning
    print("\n[Step 9] Evaluating Robustness under Weight Pruning (10%, 30%, 50%, 70%)...")
    prune_levels = [0.10, 0.30, 0.50, 0.70]
    pruning_results = {}
    for p in prune_levels:
        pruned_acc = RobustnessEvaluator.evaluate_pruning_attack(
            model=watermarked_model, triggers=norm_triggers, targets=targets, prune_ratio=p
        )
        pruning_results[f"prune_{int(p*100)}%"] = float(pruned_acc)
        print(f" -> Prune {int(p*100)}%: Trigger Accuracy = {pruned_acc * 100:.1f}%")

    # 10. Save JSON Report
    report = {
        "dataset": "MNIST (10,000 real test samples)",
        "model_architecture": "SimpleCNN (2 Conv Layers + AdaptiveAvgPool + Linear)",
        "baseline_model": {
            "accuracy": base_metrics["accuracy"],
            "macro_precision": base_metrics["macro_precision"],
            "macro_recall": base_metrics["macro_recall"],
            "macro_f1": base_metrics["macro_f1"]
        },
        "watermarked_model": {
            "accuracy": wm_metrics["accuracy"],
            "macro_precision": wm_metrics["macro_precision"],
            "macro_recall": wm_metrics["macro_recall"],
            "macro_f1": wm_metrics["macro_f1"],
            "accuracy_delta": delta_acc,
            "task_retention_rate": retention
        },
        "watermark_verification": {
            "num_triggers": verify_res["total_tested"],
            "trigger_matches": verify_res["correct_matches"],
            "trigger_accuracy": verify_res["accuracy"],
            "p_value": verify_res["p_value"],
            "verdict": verify_res["verdict"]
        },
        "counterfeit_challenge": {
            "false_alarm_matches": fake_res["correct_matches"],
            "false_alarm_rate": fake_res["accuracy"],
            "verdict": fake_res["verdict"]
        },
        "pruning_robustness": pruning_results
    }

    report_path = "./research_outputs/real_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n -> Full JSON report saved to {report_path}")

    # 11. Plot Confusion Matrix & Pruning Robustness Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix
    cm_arr = np.array(wm_metrics["confusion_matrix"])
    im = ax1.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title(f"Watermarked Model: Real Confusion Matrix (10,000 Test Images)\nAccuracy: {wm_metrics['accuracy']*100:.2f}% | Retention: {retention:.2f}%", fontsize=11, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xlabel("Predicted Class", fontsize=10)
    ax1.set_ylabel("True Class", fontsize=10)
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))

    # Pruning Curve
    prune_pcts = [0, 10, 30, 50, 70]
    prune_vals = [verify_res['accuracy'] * 100] + [pruning_results[f"prune_{p}%"] * 100 for p in [10, 30, 50, 70]]
    ax2.plot(prune_pcts, prune_vals, marker="o", color="#0E7490", lw=2.5, label="Trigger Accuracy (%)")
    ax2.axhline(y=85.0, color="#EF4444", linestyle="--", label="Ownership Verification Threshold (85%)")
    ax2.set_xlabel("Weights Pruned (%)", fontsize=10)
    ax2.set_ylabel("Trigger Accuracy (%)", fontsize=10)
    ax2.set_title("Watermark Robustness Against Weight Pruning Attack", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=10)

    plot_path = "./research_outputs/real_evaluation_metrics.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f" -> Real evaluation plots saved to {plot_path}")
    print("\nBenchmark successfully completed with 100% genuine data!")


if __name__ == "__main__":
    run_benchmark()
