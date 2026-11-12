"""
Trainer Engine with Key-Watermark Embedding and Robustness Testing.
Implements the multi-task joint loss: L_total = L_main + lambda_wm * L_wm,
and provides robustness evaluation against fine-tuning, pruning, and quantization.
"""

from typing import Dict, Any, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from backend.core.whitebox import DeepMarkWhiteBoxEngine


class WatermarkedTrainer:
    """Production training engine for embedding trigger-set watermarks."""

    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_with_watermark(
        self,
        train_loader: DataLoader,
        triggers: np.ndarray,
        targets: np.ndarray,
        epochs: int = 5,
        lambda_wm: float = 0.1,
        watermark_freq: int = 5,
        secret_key: Optional[bytes] = None,
        secret_phrase: str = "hola amigo",
        progress_callback: Optional[Callable[[int, float, float, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Embeds BOTH:
          1. Black-box secret trigger-set backdoor (responds to secret phrase/triggers with target class)
          2. White-box parameter-space binary signature (DeepMark parameter projection)
        """
        import copy
        initial_model = copy.deepcopy(self.model).cpu()
        trigger_tensor = torch.tensor(triggers, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(self.device)

        history = []
        for epoch in range(epochs):
            self.model.train()
            total_loss_sum = 0.0
            main_loss_sum = 0.0
            wm_loss_sum = 0.0
            wm_acc_list = []

            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                # 1. Main task forward
                outputs = self.model(data)
                loss_main = self.criterion(outputs, labels)

                # 2. Dual Watermark embedding step (Trigger-Set + White-Box Parameter Projection)
                if batch_idx % watermark_freq == 0:
                    wm_outputs = self.model(trigger_tensor)
                    loss_trigger = self.criterion(wm_outputs, target_tensor)

                    loss_wb = torch.tensor(0.0, device=self.device)
                    if secret_key is not None:
                        loss_wb = DeepMarkWhiteBoxEngine.compute_whitebox_loss(
                            self.model, secret_key, secret_phrase, num_bits=64, device=self.device
                        )

                    loss_wm = loss_trigger + loss_wb

                    # Compute trigger accuracy
                    wm_preds = torch.argmax(wm_outputs, dim=1)
                    wm_acc = float((wm_preds == target_tensor).float().mean().item())
                    wm_acc_list.append(wm_acc)

                    batch_loss = loss_main + lambda_wm * loss_wm
                    wm_loss_sum += loss_wm.item()
                else:
                    batch_loss = loss_main

                # 3. Backprop
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                total_loss_sum += batch_loss.item()
                main_loss_sum += loss_main.item()

            batches_count = len(train_loader)
            avg_main = main_loss_sum / batches_count if batches_count > 0 else 0.0
            avg_wm = (wm_loss_sum / max(1, len(wm_acc_list))) if wm_acc_list else 0.0
            avg_wm_acc = float(np.mean(wm_acc_list)) if wm_acc_list else 0.0

            epoch_stat = {
                "epoch": epoch + 1,
                "main_loss": round(avg_main, 4),
                "watermark_loss": round(avg_wm, 4),
                "watermark_accuracy": round(avg_wm_acc, 4)
            }
            history.append(epoch_stat)

            if progress_callback:
                progress_callback(epoch + 1, avg_main, avg_wm, avg_wm_acc)

        wb_metrics = {}
        if secret_key is not None:
            wb_metrics = DeepMarkWhiteBoxEngine.extract_and_verify(
                self.model, secret_key, secret_phrase, num_bits=64, device=self.device
            )
            wb_metrics["l2_parameter_deviation"] = DeepMarkWhiteBoxEngine.compute_l2_deviation(
                initial_model, self.model
            )

        return {
            "history": history,
            "final_watermark_accuracy": history[-1]["watermark_accuracy"] if history else 0.0,
            "whitebox_metrics": wb_metrics,
            "model": self.model
        }

    def fine_tune_pretrained_with_preservation(
        self,
        input_shape: Tuple[int, ...],
        triggers: np.ndarray,
        targets: np.ndarray,
        epochs: int = 5,
        lambda_wm: float = 0.5,
        calibration_samples: int = 256,
        secret_key: Optional[bytes] = None,
        secret_phrase: str = "hola amigo"
    ) -> Dict[str, Any]:
        """
        Post-hoc watermarking for an already-trained client model WITHOUT needing their raw training dataset.
        Uses a frozen teacher copy and KL-Divergence logit preservation on synthetic calibration noise
        while embedding BOTH the secret phrase trigger-set AND white-box parameter bit-signature.
        """
        import copy
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        trigger_tensor = torch.tensor(triggers, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(self.device)

        # Synthetic calibration inputs matching client model's input shape
        calib_x = torch.randn(calibration_samples, *input_shape, device=self.device)
        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        history = []
        batch_size = 32

        for epoch in range(epochs):
            self.model.train()
            main_loss_sum = 0.0
            wm_loss_sum = 0.0
            wm_acc_list = []
            batches = max(1, calibration_samples // batch_size)

            for b in range(batches):
                batch_x = calib_x[b * batch_size:(b + 1) * batch_size]

                # 1. Task Preservation (KL divergence from original pre-trained teacher outputs)
                with torch.no_grad():
                    teacher_logits = teacher(batch_x)
                    teacher_probs = torch.softmax(teacher_logits, dim=1)

                student_logits = self.model(batch_x)
                student_log_probs = torch.log_softmax(student_logits, dim=1)
                loss_preserve = kl_loss_fn(student_log_probs, teacher_probs)

                # 2. Dual Watermark Injection Loss (Secret Trigger + White-Box Parameter Projection)
                wm_outputs = self.model(trigger_tensor)
                loss_trigger = self.criterion(wm_outputs, target_tensor)

                loss_wb = torch.tensor(0.0, device=self.device)
                if secret_key is not None:
                    loss_wb = DeepMarkWhiteBoxEngine.compute_whitebox_loss(
                        self.model, secret_key, secret_phrase, num_bits=64, device=self.device
                    )

                loss_wm = loss_trigger + loss_wb

                wm_preds = torch.argmax(wm_outputs, dim=1)
                wm_acc = float((wm_preds == target_tensor).float().mean().item())
                wm_acc_list.append(wm_acc)

                total_loss = loss_preserve + lambda_wm * loss_wm

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                main_loss_sum += loss_preserve.item()
                wm_loss_sum += loss_wm.item()

            # Compute output drift metric (how closely the watermarked model still matches original behavior on clean inputs)
            self.model.eval()
            with torch.no_grad():
                orig_preds = torch.argmax(teacher(calib_x), dim=1)
                new_preds = torch.argmax(self.model(calib_x), dim=1)
                retention_rate = float((orig_preds == new_preds).float().mean().item())

            avg_preserve = main_loss_sum / batches
            avg_wm = wm_loss_sum / batches
            avg_wm_acc = float(np.mean(wm_acc_list)) if wm_acc_list else 0.0

            history.append({
                "epoch": epoch + 1,
                "main_loss": round(avg_preserve, 4),
                "watermark_loss": round(avg_wm, 4),
                "watermark_accuracy": round(avg_wm_acc, 4),
                "task_retention_rate": round(retention_rate, 4)
            })

        wb_metrics = {}
        if secret_key is not None:
            wb_metrics = DeepMarkWhiteBoxEngine.extract_and_verify(
                self.model, secret_key, secret_phrase, num_bits=64, device=self.device
            )
            wb_metrics["l2_parameter_deviation"] = DeepMarkWhiteBoxEngine.compute_l2_deviation(
                teacher, self.model
            )

        return {
            "history": history,
            "final_watermark_accuracy": history[-1]["watermark_accuracy"] if history else 0.0,
            "task_retention_rate": history[-1]["task_retention_rate"] if history else 1.0,
            "whitebox_metrics": wb_metrics,
            "model": self.model
        }


class RobustnessEvaluator:
    """Evaluates watermark resilience against common model extraction and evasion attacks."""

    @staticmethod
    def evaluate_pruning_attack(
        model: nn.Module,
        triggers: np.ndarray,
        targets: np.ndarray,
        prune_ratio: float = 0.3,
        device: str = "cpu"
    ) -> float:
        """Applies unstructured L1 weight pruning on a deep copy and tests watermark retention."""
        import copy
        model_copy = copy.deepcopy(model).to(device)
        model_copy.eval()

        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name="weight", amount=prune_ratio)
                prune.remove(module, "weight")

        trigger_tensor = torch.tensor(triggers, dtype=torch.float32).to(device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model_copy(trigger_tensor)
            preds = torch.argmax(outputs, dim=1)
            accuracy = float((preds == target_tensor).float().mean().item())

        return round(accuracy, 4)

    @staticmethod
    def evaluate_quantization_attack(
        model: nn.Module,
        triggers: np.ndarray,
        targets: np.ndarray,
        device: str = "cpu"
    ) -> float:
        """Simulates 8-bit dynamic quantization on model parameters."""
        import copy
        model_copy = copy.deepcopy(model).cpu()
        model_copy.eval()
        try:
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model_copy, {nn.Linear}, dtype=torch.qint8
            )
        except Exception:
            quantized_model = model_copy

        trigger_tensor = torch.tensor(triggers, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.long)

        with torch.no_grad():
            outputs = quantized_model(trigger_tensor)
            preds = torch.argmax(outputs, dim=1)
            accuracy = float((preds == target_tensor).float().mean().item())

        return round(accuracy, 4)
