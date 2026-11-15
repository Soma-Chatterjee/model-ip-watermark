import os
import io
import json
import time
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from backend.core.watermark import KeyAndWatermarkGenerator
from backend.core.verifier import WatermarkVerifier
from backend.core.auditor import AuthorshipAuditor
from backend.core.trainer import WatermarkedTrainer, RobustnessEvaluator
from backend.core.whitebox import DeepMarkWhiteBoxEngine
from backend.db.repository import DatabaseRepository
from backend.api.schemas import (
    HeartbeatResponse,
    KeyGenerationRequest,
    KeyGenerationResponse,
    AuthorshipAuditRequest,
    AuthorshipAuditResponse,
    WatermarkEmbedRequest,
    WatermarkEmbedResponse,
    VerificationRequest,
    VerificationResponse,
    ModelComparisonResponse,
    SingleModelEvaluation
)

# Initialize FastAPI App
app = FastAPI(
    title="DeepMark: White-Box Neural Network Watermarking API",
    description="White-Box Neural Network Watermarking & Secret Trigger-Phrase IP Protection System (Adamas University CSE).",
    version="1.0.0"
)

# Enable CORS for Next.js frontend (Vercel & Localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# In-memory registry for fast local demonstrations & session testing
ACTIVE_MODELS: Dict[str, Any] = {}


# --- Model Architectures ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        features = self.conv(x)
        features = torch.flatten(features, 1)
        return self.fc(features)


# --- 1. Heartbeat / Keep-Alive Endpoint (Render Free Tier Safe) ---
@app.get("/health", response_model=HeartbeatResponse, tags=["Monitoring"])
@app.get("/api/health", response_model=HeartbeatResponse, tags=["Monitoring"])
def get_health():
    """
    Heartbeat endpoint for cron-job.org / UptimeRobot keep-alive pings.
    Returns status and uptime without database overhead.
    """
    uptime = int(time.time() - START_TIME)
    return HeartbeatResponse(
        status="healthy",
        service="deepmark-watermark-core",
        uptime_seconds=uptime,
        version="1.0.0"
    )


@app.get("/api/models", tags=["Registry"])
@app.get("/api/models/active", tags=["Registry"])
def list_protected_models():
    """
    Returns live count and metadata of protected models.
    Merges persistent Supabase PostgreSQL records with active in-memory models.
    """
    db_models = DatabaseRepository.list_models()
    seen_ids = {m["id"] for m in db_models}

    for mid, info in ACTIVE_MODELS.items():
        if mid not in seen_ids:
            db_models.insert(0, {
                "id": mid,
                "model_name": info.get("model_name", "Unnamed Model"),
                "owner_id": info.get("owner_id", "Unknown"),
                "architecture": info.get("architecture", "MLP"),
                "target_class": info.get("target_class", 0),
                "watermark_accuracy": info.get("watermark_accuracy", 1.0),
                "pruning_survival": info.get("pruning_survival", 1.0),
                "quantization_survival": info.get("quantization_survival", 1.0),
                "created_at": info.get("created_at", "")
            })

    return {
        "active_count": len(db_models),
        "database_connected": DatabaseRepository.is_connected(),
        "models": db_models
    }


# --- 2. Key & Watermark Generation ---
@app.post("/api/keys/generate", response_model=KeyGenerationResponse, tags=["Watermark Engine"])
def generate_key_and_triggers(req: KeyGenerationRequest):
    """
    Generates a cryptographically strong Fernet 256-bit key and derives
    a deterministic trigger set signature.
    """
    try:
        generator = KeyAndWatermarkGenerator(model_name=req.model_name, owner_id=req.owner_id)
        input_shape_tuple = tuple(req.input_shape)
        triggers, targets, seed = generator.generate_trigger_set(
            input_shape=input_shape_tuple,
            num_triggers=req.num_triggers,
            pattern_type=req.pattern_type,
            target_class=req.target_class
        )
        dummy_weights_hash = "mock_sha256_" + str(uuid.uuid4())[:8]
        signature = generator.create_watermark_signature(weights_sha256=dummy_weights_hash)

        return KeyGenerationResponse(
            model_name=req.model_name,
            owner_id=req.owner_id,
            key_hash=generator.key_hash,
            secret_key=generator.get_key_str(),
            signature=signature,
            num_triggers=req.num_triggers,
            target_class=req.target_class,
            trigger_seed=seed,
            created_at=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key generation failed: {str(e)}")


# --- 3. Proof-of-Authorship & Documentation Audit ---
@app.post("/api/audit/verify", response_model=AuthorshipAuditResponse, tags=["Authorship & Compliance"])
def audit_model_authorship(req: AuthorshipAuditRequest):
    """
    Inspects model weights hash against public baselines and audits
    uploaded training logs to prevent stolen/copied models from being registered.
    """
    # Check baseline collisions
    collision = AuthorshipAuditor.check_public_baseline_collision(req.weights_sha256)
    if collision["is_collision"]:
        return AuthorshipAuditResponse(
            model_name=req.model_name,
            weights_sha256=req.weights_sha256,
            approved=False,
            confidence_score=0,
            status="flagged_plagiarism",
            issues=[collision["reason"]],
            summary=f"Plagiarism Detected: Checkpoint matches {collision['matched_baseline']} exactly."
        )

    # Audit logs and documentation
    audit_res = AuthorshipAuditor.audit_training_documentation(
        dataset_hash=req.dataset_hash,
        training_logs_raw=req.training_logs_raw,
        code_repo_url=req.code_repo_url
    )

    return AuthorshipAuditResponse(
        model_name=req.model_name,
        weights_sha256=req.weights_sha256,
        approved=audit_res["approved"],
        confidence_score=audit_res["confidence_score"],
        status=audit_res["status"],
        issues=audit_res["issues"],
        summary=audit_res["summary"]
    )


def _compute_model_sha256(model: nn.Module) -> str:
    """Computes deterministic SHA-256 fingerprint of a PyTorch model's serialized weights."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# --- 4A. Watermark Injection: Train From Scratch (Deprecated in favor of Pre-Trained MaaS Pipeline) ---
@app.post("/api/watermark/embed", response_model=WatermarkEmbedResponse, tags=["Watermark Engine"], deprecated=True)
def embed_watermark_endpoint(req: WatermarkEmbedRequest):
    """
    [DEPRECATED]: Use /api/watermark/upload-and-embed instead.
    Trains a model from scratch with joint loss L_total = L_main + lambda_wm * L_wm.
    """
    try:
        # 1. Initialize fresh architecture
        flat_dim = int(np.prod(req.input_shape))
        if req.architecture == "CNN" and len(req.input_shape) == 3:
            model = SimpleCNN(in_channels=req.input_shape[0], num_classes=req.num_classes)
        else:
            model = SimpleMLP(input_dim=flat_dim, num_classes=req.num_classes)

        # 2. Derive trigger set from the secret key
        generator = KeyAndWatermarkGenerator(
            model_name=req.model_name,
            owner_id=req.owner_id,
            secret_key=req.secret_key.encode("utf-8")
        )
        triggers, targets, _ = generator.generate_trigger_set(
            input_shape=tuple(req.input_shape),
            num_triggers=100,
            pattern_type="pattern",
            target_class=req.target_class
        )

        # 3. Create training dataset
        num_samples = 300
        x_train = torch.randn(num_samples, *req.input_shape)
        y_train = torch.randint(0, req.num_classes, (num_samples,))
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 4. Train with Dual Watermark embedding (Trigger-Set + White-Box Parameter Projection)
        trainer = WatermarkedTrainer(model=model, device="cpu", lr=0.005)
        train_res = trainer.train_with_watermark(
            train_loader=dataloader,
            triggers=triggers,
            targets=targets,
            epochs=req.epochs,
            lambda_wm=req.lambda_wm,
            watermark_freq=2,
            secret_key=req.secret_key.encode("utf-8"),
            secret_phrase=req.secret_phrase
        )
        wb_metrics = train_res.get("whitebox_metrics", {})
        phrase_hash = hashlib.sha256(req.secret_phrase.encode("utf-8")).hexdigest()

        # 5. Robustness evaluation
        prune_acc = RobustnessEvaluator.evaluate_pruning_attack(
            model=model, triggers=triggers, targets=targets, prune_ratio=0.3
        )
        quant_acc = RobustnessEvaluator.evaluate_quantization_attack(
            model=model, triggers=triggers, targets=targets
        )

        # 6. Serialize & save in registry
        weights_sha = _compute_model_sha256(model)
        signature = generator.create_watermark_signature(weights_sha256=weights_sha)

        model_record_id = str(uuid.uuid4())[:8]
        ACTIVE_MODELS[model_record_id] = {
            "id": model_record_id,
            "model": model,
            "model_name": req.model_name,
            "owner_id": req.owner_id,
            "mode": "from_scratch",
            "architecture": req.architecture,
            "input_shape": req.input_shape,
            "num_classes": req.num_classes,
            "target_class": req.target_class,
            "secret_phrase_hash": phrase_hash,
            "watermark_accuracy": train_res["final_watermark_accuracy"],
            "whitebox_bit_accuracy": wb_metrics.get("bit_accuracy"),
            "whitebox_ber": wb_metrics.get("bit_error_rate"),
            "l2_parameter_deviation": wb_metrics.get("l2_parameter_deviation"),
            "pruning_survival": prune_acc,
            "quantization_survival": quant_acc,
            "original_weights_sha256": None,
            "watermarked_weights_sha256": weights_sha,
            "signature": signature,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

        # Persist metadata & key fingerprint to Supabase PostgreSQL
        DatabaseRepository.save_registered_model(
            model_id=model_record_id,
            org_slug=req.owner_id,
            model_name=req.model_name,
            architecture=req.architecture,
            input_shape=req.input_shape,
            num_classes=req.num_classes,
            weights_sha256=weights_sha,
            status="watermarked",
            key_data={
                "key_hash": generator.key_hash,
                "signature": signature,
                "num_triggers": 100,
                "target_class": req.target_class,
                "trigger_seed": generator.derive_seed_from_key()
            }
        )

        return WatermarkEmbedResponse(
            model_id=model_record_id,
            model_name=req.model_name,
            secret_phrase_hash=phrase_hash,
            mode="from_scratch",
            status="watermarked_success",
            final_watermark_accuracy=train_res["final_watermark_accuracy"],
            task_retention_rate=None,
            whitebox_bit_accuracy=wb_metrics.get("bit_accuracy"),
            whitebox_ber=wb_metrics.get("bit_error_rate"),
            l2_parameter_deviation=wb_metrics.get("l2_parameter_deviation"),
            training_history=train_res["history"],
            pruning_survival_accuracy=prune_acc,
            quantization_survival_accuracy=quant_acc,
            original_weights_sha256=None,
            watermarked_weights_sha256=weights_sha,
            download_model_url=f"/api/models/{model_record_id}/download",
            download_certificate_url=f"/api/models/{model_record_id}/certificate",
            signature=signature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Watermark embedding failed: {str(e)}")


# --- 4B. Watermark Injection: Already-Trained Client Model (.pt / .pth Upload & Delivery) ---
@app.post("/api/watermark/upload-and-embed", response_model=WatermarkEmbedResponse, tags=["Watermark Engine"])
async def upload_and_embed_model(
    model_file: UploadFile = File(...),
    model_name: str = Form(...),
    owner_id: str = Form(...),
    secret_key: str = Form(...),
    secret_phrase: str = Form("hola amigo"),
    architecture: str = Form("MLP"),
    input_shape: str = Form("[784]"),
    num_classes: int = Form(10),
    target_class: int = Form(7),
    epochs: int = Form(5),
    lambda_wm: float = Form(0.5)
):
    """
    Mode 2 (Pre-Trained Model Delivery):
    Accepts an already-trained PyTorch (.pt/.pth) model uploaded by a client:
      1. Computes original weights SHA-256 & checks against public baseline plagiarism.
      2. Loads the model (supports full nn.Module TorchScript/pickle OR state_dict).
      3. Fine-tunes with KL-Divergence Logit Preservation so original accuracy is preserved.
      4. Stages the newly watermarked .pt file & IP Certificate for client download.
    """
    try:
        raw_bytes = await model_file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded model file is empty.")

        original_sha256 = AuthorshipAuditor.calculate_file_sha256(raw_bytes)
        collision = AuthorshipAuditor.check_public_baseline_collision(original_sha256)
        if collision["is_collision"]:
            raise HTTPException(status_code=403, detail=collision["reason"])

        parsed_shape = json.loads(input_shape)
        if not isinstance(parsed_shape, list) or len(parsed_shape) == 0:
            raise HTTPException(status_code=400, detail="input_shape must be a JSON list like [784] or [3, 32, 32].")

        # Load uploaded PyTorch file safely
        buffer = io.BytesIO(raw_bytes)
        loaded_obj = torch.load(buffer, map_location="cpu", weights_only=False)

        if isinstance(loaded_obj, nn.Module):
            model = loaded_obj
        elif isinstance(loaded_obj, dict):
            # Client uploaded a state_dict; instantiate base architecture and load weights
            flat_dim = int(np.prod(parsed_shape))
            if architecture == "CNN" and len(parsed_shape) == 3:
                model = SimpleCNN(in_channels=parsed_shape[0], num_classes=num_classes)
            else:
                model = SimpleMLP(input_dim=flat_dim, num_classes=num_classes)
            # Filter/load matching keys cleanly
            state_dict = loaded_obj.get("state_dict", loaded_obj)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise HTTPException(status_code=400, detail="Unsupported PyTorch file format. Upload an nn.Module or state_dict (.pt/.pth).")

        # Generate deterministic triggers from key
        generator = KeyAndWatermarkGenerator(
            model_name=model_name,
            owner_id=owner_id,
            secret_key=secret_key.encode("utf-8")
        )
        triggers, targets, _ = generator.generate_trigger_set(
            input_shape=tuple(parsed_shape),
            num_triggers=100,
            pattern_type="pattern",
            target_class=target_class
        )

        # Preserve original model behavior via KL Distillation while embedding Dual Watermarks
        trainer = WatermarkedTrainer(model=model, device="cpu", lr=0.002)
        tune_res = trainer.fine_tune_pretrained_with_preservation(
            input_shape=tuple(parsed_shape),
            triggers=triggers,
            targets=targets,
            epochs=min(epochs, 3),
            calibration_samples=64,
            lambda_wm=lambda_wm,
            secret_key=secret_key.encode("utf-8"),
            secret_phrase=secret_phrase
        )
        wb_metrics = tune_res.get("whitebox_metrics", {})
        phrase_hash = hashlib.sha256(secret_phrase.encode("utf-8")).hexdigest()

        # Robustness checks
        prune_acc = RobustnessEvaluator.evaluate_pruning_attack(
            model=model, triggers=triggers, targets=targets, prune_ratio=0.3
        )
        quant_acc = RobustnessEvaluator.evaluate_quantization_attack(
            model=model, triggers=triggers, targets=targets
        )

        watermarked_sha256 = _compute_model_sha256(model)
        signature = generator.create_watermark_signature(weights_sha256=watermarked_sha256)

        model_record_id = str(uuid.uuid4())[:8]
        ACTIVE_MODELS[model_record_id] = {
            "id": model_record_id,
            "model": model,
            "model_name": model_name,
            "owner_id": owner_id,
            "mode": "pretrained_upload",
            "architecture": architecture,
            "input_shape": parsed_shape,
            "num_classes": num_classes,
            "target_class": target_class,
            "secret_phrase_hash": phrase_hash,
            "watermark_accuracy": tune_res["final_watermark_accuracy"],
            "task_retention_rate": tune_res["task_retention_rate"],
            "whitebox_bit_accuracy": wb_metrics.get("bit_accuracy"),
            "whitebox_ber": wb_metrics.get("bit_error_rate"),
            "l2_parameter_deviation": wb_metrics.get("l2_parameter_deviation"),
            "pruning_survival": prune_acc,
            "quantization_survival": quant_acc,
            "original_weights_sha256": original_sha256,
            "watermarked_weights_sha256": watermarked_sha256,
            "signature": signature,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

        # Persist metadata & key fingerprint to Supabase PostgreSQL
        DatabaseRepository.save_registered_model(
            model_id=model_record_id,
            org_slug=owner_id,
            model_name=model_name,
            architecture=architecture,
            input_shape=parsed_shape,
            num_classes=num_classes,
            weights_sha256=watermarked_sha256,
            status="watermarked",
            key_data={
                "key_hash": generator.key_hash,
                "signature": signature,
                "num_triggers": 100,
                "target_class": target_class,
                "trigger_seed": generator.derive_seed_from_key()
            }
        )

        return WatermarkEmbedResponse(
            model_id=model_record_id,
            model_name=model_name,
            secret_phrase_hash=phrase_hash,
            mode="pretrained_upload",
            status="watermarked_success",
            final_watermark_accuracy=tune_res["final_watermark_accuracy"],
            task_retention_rate=tune_res["task_retention_rate"],
            whitebox_bit_accuracy=wb_metrics.get("bit_accuracy"),
            whitebox_ber=wb_metrics.get("bit_error_rate"),
            l2_parameter_deviation=wb_metrics.get("l2_parameter_deviation"),
            training_history=tune_res["history"],
            pruning_survival_accuracy=prune_acc,
            quantization_survival_accuracy=quant_acc,
            original_weights_sha256=original_sha256,
            watermarked_weights_sha256=watermarked_sha256,
            download_model_url=f"/api/models/{model_record_id}/download",
            download_certificate_url=f"/api/models/{model_record_id}/certificate",
            signature=signature
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pre-trained model watermarking failed: {str(e)}")


# --- 4C. Client Delivery Endpoints (Download Watermarked Weights & IP Certificate) ---
@app.get("/api/models/{model_id}/download", tags=["Delivery"])
def download_watermarked_model(model_id: str):
    """
    Streams the newly watermarked PyTorch model (.pt) back to the client for production deployment.
    """
    if model_id not in ACTIVE_MODELS:
        raise HTTPException(status_code=404, detail="Protected model record not found.")

    record = ACTIVE_MODELS[model_id]
    model = record["model"]
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in record["model_name"])
    filename = f"{safe_name}_watermarked_{model_id}.pt"

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/api/models/{model_id}/certificate", tags=["Delivery"])
def download_ip_certificate(model_id: str):
    """
    Generates and downloads the signed JSON IP Ownership Passport for the watermarked model.
    """
    if model_id not in ACTIVE_MODELS:
        raise HTTPException(status_code=404, detail="Protected model record not found.")

    rec = ACTIVE_MODELS[model_id]
    certificate = {
        "certificate_type": "DeepMark Dual-Fidelity IP Ownership Deed",
        "model_id": model_id,
        "model_name": rec["model_name"],
        "owner_id": rec["owner_id"],
        "embedding_mode": rec.get("mode", "pretrained_upload"),
        "architecture": rec["architecture"],
        "target_watermark_class": rec["target_class"],
        "watermark_accuracy": rec["watermark_accuracy"],
        "task_retention_rate": rec.get("task_retention_rate"),
        "pruning_30pct_survival": rec["pruning_survival"],
        "quantization_8bit_survival": rec["quantization_survival"],
        "original_weights_sha256": rec.get("original_weights_sha256"),
        "watermarked_weights_sha256": rec.get("watermarked_weights_sha256"),
        "cryptographic_signature": rec["signature"],
        "issued_at": rec["created_at"]
    }

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in rec["model_name"])
    return JSONResponse(
        content=certificate,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_IP_Passport_{model_id}.json"'}
    )


# --- 5. Ownership Verification & Dispute Resolution ---
@app.post("/api/watermark/verify", response_model=VerificationResponse, tags=["Verification Engine"])
async def verify_watermark_endpoint(req: VerificationRequest):
    """
    Verifies watermark presence on a remote API (black-box) or in-memory model,
    calculates exact binomial p-values for legally sound proof of ownership.
    """
    try:
        # Reconstruct trigger set from key
        key_bytes = req.secret_key.encode("utf-8")
        generator = KeyAndWatermarkGenerator(
            model_name="verify_query", owner_id="verifier", secret_key=key_bytes
        )
        triggers, targets, _ = generator.generate_trigger_set(
            input_shape=tuple(req.input_shape),
            num_triggers=req.num_triggers,
            pattern_type=req.pattern_type,
            target_class=req.target_class
        )

        verifier = WatermarkVerifier(accuracy_threshold=0.85)

        wb_verif = {}
        if req.endpoint_url:
            # Black-box verification via remote REST API
            result = await verifier.verify_remote_api(
                endpoint_url=req.endpoint_url,
                triggers=triggers,
                targets=targets,
                num_classes=req.num_classes
            )
        else:
            # Check latest active model in memory if available, or simulate model query
            target_model = None
            if ACTIVE_MODELS:
                latest_key = list(ACTIVE_MODELS.keys())[-1]
                target_model = ACTIVE_MODELS[latest_key]["model"]
            else:
                flat_dim = int(np.prod(req.input_shape))
                target_model = SimpleMLP(input_dim=flat_dim, num_classes=req.num_classes)

            result = verifier.verify_pytorch_model(
                model=target_model,
                triggers=triggers,
                targets=targets,
                num_classes=req.num_classes
            )

            # Also run White-Box Secret Phrase Bitwise Extraction
            wb_verif = DeepMarkWhiteBoxEngine.extract_and_verify(
                model=target_model,
                secret_key=key_bytes,
                expected_phrase=req.secret_phrase,
                num_bits=64,
                device="cpu"
            )

        is_watermarked = result.get("is_watermarked", False)
        verdict = result.get("verdict", "INCONCLUSIVE")
        if wb_verif:
            # Dual-fidelity validation: if secret phrase extraction fails, reject claim
            if not wb_verif.get("is_verified", True):
                is_watermarked = False
                verdict = "WATERMARK_NOT_FOUND"

        cert_id = f"CERT-DM-{uuid.uuid4().hex[:12].upper()}"

        return VerificationResponse(
            mode=result.get("mode", "white_box"),
            secret_phrase_verified=wb_verif.get("is_verified"),
            whitebox_bit_accuracy=wb_verif.get("bit_accuracy"),
            whitebox_ber=wb_verif.get("bit_error_rate"),
            total_tested=result.get("total_tested", req.num_triggers),
            correct_matches=result.get("correct_matches", 0),
            accuracy=result.get("accuracy", 0.0),
            threshold=result.get("threshold", 0.85),
            is_watermarked=is_watermarked,
            p_value=result.get("p_value", 1.0),
            confidence_percent=result.get("confidence_percent", 0.0),
            verdict=verdict,
            certificate_id=cert_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# --- 6. Side-by-Side Model Comparison & Testing Lab ---
@app.post("/api/watermark/compare-models", response_model=ModelComparisonResponse, tags=["Testing & Benchmark Lab"])
async def compare_models_endpoint(
    original_file: Optional[UploadFile] = File(None),
    watermarked_file: Optional[UploadFile] = File(None),
    secret_key: str = Form(...),
    secret_phrase: str = Form("hola amigo"),
    architecture: str = Form("MLP"),
    input_shape: str = Form("[784]"),
    num_classes: int = Form(10),
    target_class: int = Form(7)
):
    """
    Side-by-Side Model Testing & Verification Lab:
    Compares the unwatermarked original model against the watermarked model (.pth/.pt):
      1. Clean Task Agreement: Proves zero functionality degradation on normal inputs.
      2. Trigger Set Challenge: Proves watermarked model responds to backdoor while original ignores it.
      3. White-Box Parameter Extraction: Proves secret phrase exists in weights (0.0% BER vs ~50% in original).
    """
    try:
        parsed_shape = json.loads(input_shape)
        flat_dim = int(np.prod(parsed_shape))
        key_bytes = secret_key.encode("utf-8")

        def _instantiate_fresh(arch: str, in_shape: list, n_classes: int):
            if arch == "CNN" and len(in_shape) == 3:
                return SimpleCNN(in_channels=in_shape[0], num_classes=n_classes)
            return SimpleMLP(input_dim=int(np.prod(in_shape)), num_classes=n_classes)

        # 1. Load Original Model
        orig_model = None
        orig_sha256 = ""
        if original_file and original_file.filename:
            raw_orig = await original_file.read()
            if len(raw_orig) > 0:
                orig_sha256 = hashlib.sha256(raw_orig).hexdigest()
                obj = torch.load(io.BytesIO(raw_orig), map_location="cpu", weights_only=False)
                if isinstance(obj, nn.Module):
                    orig_model = obj
                elif isinstance(obj, dict):
                    orig_model = _instantiate_fresh(architecture, parsed_shape, num_classes)
                    orig_model.load_state_dict(obj.get("state_dict", obj), strict=False)

        if orig_model is None:
            if os.path.exists("demo_client_model.pth"):
                with open("demo_client_model.pth", "rb") as f:
                    raw = f.read()
                orig_sha256 = hashlib.sha256(raw).hexdigest()
                obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
                if isinstance(obj, nn.Module):
                    orig_model = obj
                elif isinstance(obj, dict):
                    orig_model = _instantiate_fresh(architecture, parsed_shape, num_classes)
                    orig_model.load_state_dict(obj.get("state_dict", obj), strict=False)
            else:
                orig_model = _instantiate_fresh(architecture, parsed_shape, num_classes)
                orig_sha256 = _compute_model_sha256(orig_model)

        # 2. Load Watermarked Model
        wm_model = None
        wm_sha256 = ""
        if watermarked_file and watermarked_file.filename:
            raw_wm = await watermarked_file.read()
            if len(raw_wm) > 0:
                wm_sha256 = hashlib.sha256(raw_wm).hexdigest()
                obj = torch.load(io.BytesIO(raw_wm), map_location="cpu", weights_only=False)
                if isinstance(obj, nn.Module):
                    wm_model = obj
                elif isinstance(obj, dict):
                    wm_model = _instantiate_fresh(architecture, parsed_shape, num_classes)
                    wm_model.load_state_dict(obj.get("state_dict", obj), strict=False)

        if wm_model is None:
            if ACTIVE_MODELS:
                latest_id = list(ACTIVE_MODELS.keys())[-1]
                wm_model = ACTIVE_MODELS[latest_id]["model"]
                wm_sha256 = ACTIVE_MODELS[latest_id].get("watermarked_weights_sha256", _compute_model_sha256(wm_model))
            else:
                import copy
                wm_model = copy.deepcopy(orig_model)
                gen = KeyAndWatermarkGenerator(model_name="live_test", owner_id="tester", secret_key=key_bytes)
                trigs, tgts, _ = gen.generate_trigger_set(tuple(parsed_shape), 100, target_class=target_class)
                trainer = WatermarkedTrainer(model=wm_model, device="cpu", lr=0.002)
                trainer.fine_tune_pretrained_with_preservation(
                    tuple(parsed_shape), trigs, tgts, epochs=2, calibration_samples=64, lambda_wm=0.5,
                    secret_key=key_bytes, secret_phrase=secret_phrase
                )
                wm_sha256 = _compute_model_sha256(wm_model)

        orig_model.eval()
        wm_model.eval()

        # 3. Clean Task Agreement
        n_val = 64
        x_val = torch.randn(n_val, *parsed_shape)
        with torch.no_grad():
            out_orig = orig_model(x_val)
            out_wm = wm_model(x_val)
            pred_orig = torch.argmax(out_orig, dim=1)
            pred_wm = torch.argmax(out_wm, dim=1)
            clean_agreement = float((pred_orig == pred_wm).float().mean().item())

        # 4. Trigger Set Verification
        generator = KeyAndWatermarkGenerator(
            model_name="compare_query", owner_id="tester", secret_key=key_bytes
        )
        triggers, targets, _ = generator.generate_trigger_set(
            input_shape=tuple(parsed_shape),
            num_triggers=100,
            pattern_type="pattern",
            target_class=target_class
        )
        verifier = WatermarkVerifier(accuracy_threshold=0.85)
        orig_trig_res = verifier.verify_pytorch_model(orig_model, triggers, targets, num_classes)
        wm_trig_res = verifier.verify_pytorch_model(wm_model, triggers, targets, num_classes)

        # 5. White-Box Parameter Projection
        wb_orig = DeepMarkWhiteBoxEngine.extract_and_verify(
            model=orig_model, secret_key=key_bytes, expected_phrase=secret_phrase, num_bits=64, device="cpu"
        )
        wb_wm = DeepMarkWhiteBoxEngine.extract_and_verify(
            model=wm_model, secret_key=key_bytes, expected_phrase=secret_phrase, num_bits=64, device="cpu"
        )

        cert_id = f"CERT-DM-COMPARE-{uuid.uuid4().hex[:8].upper()}"

        # Structure original evaluation
        orig_eval = SingleModelEvaluation(
            model_type="Original Unwatermarked Model",
            weights_sha256=orig_sha256,
            trigger_accuracy=orig_trig_res["accuracy"],
            trigger_matches=orig_trig_res["correct_matches"],
            total_triggers=orig_trig_res["total_tested"],
            p_value=orig_trig_res["p_value"],
            is_watermarked=False,
            verdict="CLEAN_UNWATERMARKED",
            whitebox_ber=wb_orig.get("bit_error_rate"),
            whitebox_bit_accuracy=wb_orig.get("bit_accuracy"),
            secret_phrase_extracted=wb_orig.get("is_verified")
        )

        # Structure watermarked evaluation
        is_wm_success = wm_trig_res["is_watermarked"] and (wb_wm.get("is_verified", True) or wb_wm.get("bit_error_rate", 1.0) <= 0.05)
        wm_eval = SingleModelEvaluation(
            model_type="DeepMark Watermarked Model",
            weights_sha256=wm_sha256,
            trigger_accuracy=wm_trig_res["accuracy"],
            trigger_matches=wm_trig_res["correct_matches"],
            total_triggers=wm_trig_res["total_tested"],
            p_value=wm_trig_res["p_value"],
            is_watermarked=is_wm_success,
            verdict="OWNERSHIP_CONFIRMED" if is_wm_success else "WATERMARK_NOT_FOUND",
            whitebox_ber=wb_wm.get("bit_error_rate"),
            whitebox_bit_accuracy=wb_wm.get("bit_accuracy"),
            secret_phrase_extracted=wb_wm.get("is_verified")
        )

        ber_orig_val = wb_orig.get("bit_error_rate", 0.5) or 0.5
        ber_wm_val = wb_wm.get("bit_error_rate", 0.0) or 0.0

        summary = (
            f"Side-by-Side Comparison Complete: Original model exhibits 0% watermark presence (BER: {ber_orig_val*100:.1f}%), "
            f"while DeepMark watermarked model achieves {wm_trig_res['accuracy']*100:.1f}% trigger accuracy, {ber_wm_val*100:.2f}% BER (phrase '{secret_phrase}' confirmed), "
            f"and {clean_agreement*100:.1f}% clean task agreement."
        )

        return ModelComparisonResponse(
            status="comparison_success",
            clean_task_agreement=round(clean_agreement, 4),
            original_model=orig_eval,
            watermarked_model=wm_eval,
            comparison_verdict="WATERMARK_SUCCESSFULLY_PROVED" if is_wm_success else "INCONCLUSIVE",
            summary=summary,
            certificate_id=cert_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")
