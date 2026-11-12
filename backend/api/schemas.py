from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HeartbeatResponse(BaseModel):
    status: str = "healthy"
    service: str = "model-ip-watermark-core"
    uptime_seconds: int
    version: str = "1.0.0"


class KeyGenerationRequest(BaseModel):
    model_name: str = Field(..., example="ResNet50_Classifier")
    owner_id: str = Field(..., example="org_safeguard_ai")
    input_shape: List[int] = Field(..., example=[3, 32, 32])
    num_triggers: int = Field(default=100, ge=10, le=500)
    target_class: int = Field(default=0, ge=0)
    pattern_type: str = Field(default="pattern", pattern="^(pattern|random|semantic)$")


class KeyGenerationResponse(BaseModel):
    model_name: str
    owner_id: str
    key_hash: str
    secret_key: str
    signature: str
    num_triggers: int
    target_class: int
    trigger_seed: int
    created_at: str


class AuthorshipAuditRequest(BaseModel):
    model_name: str
    weights_sha256: str
    dataset_hash: str
    training_logs_raw: Optional[str] = None
    code_repo_url: Optional[str] = None


class AuthorshipAuditResponse(BaseModel):
    model_name: str
    weights_sha256: str
    approved: bool
    confidence_score: int
    status: str
    issues: List[str]
    summary: str


class WatermarkEmbedRequest(BaseModel):
    model_name: str
    owner_id: str
    secret_key: str
    secret_phrase: str = "hola amigo"
    architecture: str = "MLP" # MLP, CNN
    input_shape: List[int] = [784]
    num_classes: int = 10
    target_class: int = 0
    epochs: int = 5
    lambda_wm: float = 0.1


class WatermarkEmbedResponse(BaseModel):
    model_id: str
    model_name: str
    secret_phrase_hash: str
    mode: str
    status: str
    final_watermark_accuracy: float
    task_retention_rate: Optional[float] = None
    whitebox_bit_accuracy: Optional[float] = None
    whitebox_ber: Optional[float] = None
    l2_parameter_deviation: Optional[float] = None
    training_history: List[Dict[str, Any]]
    pruning_survival_accuracy: float
    quantization_survival_accuracy: float
    original_weights_sha256: Optional[str] = None
    watermarked_weights_sha256: str
    download_model_url: str
    download_certificate_url: str
    signature: str


class VerificationRequest(BaseModel):
    secret_key: str
    secret_phrase: str = "hola amigo"
    input_shape: List[int]
    num_classes: int
    target_class: int
    num_triggers: int = 100
    pattern_type: str = "pattern"
    # Either endpoint_url (black box) or target model architecture/weights
    endpoint_url: Optional[str] = None


class VerificationResponse(BaseModel):
    mode: str
    secret_phrase_verified: Optional[bool] = None
    whitebox_bit_accuracy: Optional[float] = None
    whitebox_ber: Optional[float] = None
    total_tested: int
    correct_matches: int
    accuracy: float
    threshold: float
    is_watermarked: bool
    p_value: float
    confidence_percent: float
    verdict: str
    certificate_id: str
