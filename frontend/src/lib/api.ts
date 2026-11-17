const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface KeyGenResponse {
  model_name: string;
  owner_id: string;
  key_hash: string;
  secret_key: string;
  signature: string;
  num_triggers: number;
  target_class: number;
  trigger_seed: number;
  created_at: string;
}

export interface AuditResponse {
  model_name: string;
  weights_sha256: string;
  approved: boolean;
  confidence_score: number;
  status: string;
  issues: string[];
  summary: string;
}

export interface EmbedResponse {
  model_id: string;
  model_name: string;
  secret_phrase_hash: string;
  mode: string;
  status: string;
  final_watermark_accuracy: number;
  task_retention_rate?: number | null;
  training_history: Array<{
    epoch: number;
    main_loss: number;
    watermark_loss: number;
    watermark_accuracy: number;
    task_retention_rate?: number;
  }>;
  pruning_survival_accuracy: number;
  quantization_survival_accuracy: number;
  original_weights_sha256?: string | null;
  watermarked_weights_sha256: string;
  download_model_url: string;
  download_certificate_url: string;
  signature: string;
}

export interface VerifyResponse {
  mode: string;
  total_tested: number;
  correct_matches: number;
  accuracy: number;
  threshold: number;
  is_watermarked: boolean;
  p_value: number;
  confidence_percent: number;
  verdict: string;
  certificate_id: string;
}

export interface SingleModelEval {
  model_type: string;
  weights_sha256: string;
  trigger_accuracy: number;
  trigger_matches: number;
  total_triggers: number;
  p_value: number;
  is_watermarked: boolean;
  verdict: string;
}

export interface ModelComparisonResponse {
  status: string;
  clean_task_agreement: number;
  original_model: SingleModelEval;
  watermarked_model: SingleModelEval;
  comparison_verdict: string;
  summary: string;
  certificate_id: string;
}

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export interface ProtectedModelInfo {
  id: string;
  model_name: string;
  owner_id: string;
  architecture: string;
  target_class: number;
  clean_accuracy?: number;
  watermark_accuracy: number;
  pruning_survival: number;
  quantization_survival: number;
  weights_sha256?: string;
  created_at: string;
}

export async function fetchProtectedModels(): Promise<{ active_count: number; models: ProtectedModelInfo[] }> {
  try {
    const res = await fetch(`${API_BASE}/api/models`, { cache: "no-store" });
    if (!res.ok) return { active_count: 0, models: [] };
    return await res.json();
  } catch {
    return { active_count: 0, models: [] };
  }
}

export async function generateWatermarkKey(data: {
  model_name: string;
  owner_id: string;
  input_shape: number[];
  num_triggers: number;
  target_class: number;
  pattern_type: string;
}): Promise<KeyGenResponse> {
  const res = await fetch(`${API_BASE}/api/keys/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to generate cryptographic key");
  return await res.json();
}

export async function verifyAuthorship(data: {
  model_name: string;
  weights_sha256: string;
  dataset_hash: string;
  training_logs_raw?: string;
  code_repo_url?: string;
}): Promise<AuditResponse> {
  const res = await fetch(`${API_BASE}/api/audit/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Audit verification request failed");
  return await res.json();
}

export async function embedWatermark(data: {
  model_name: string;
  owner_id: string;
  secret_key: string;
  secret_phrase: string;
  architecture: string;
  input_shape: number[];
  num_classes: number;
  target_class: number;
  epochs: number;
  lambda_wm: number;
}): Promise<EmbedResponse> {
  const res = await fetch(`${API_BASE}/api/watermark/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Watermark embedding failed");
  return await res.json();
}

export async function uploadAndEmbedModel(formData: FormData): Promise<EmbedResponse> {
  const res = await fetch(`${API_BASE}/api/watermark/upload-and-embed`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Pre-trained model watermarking failed" }));
    throw new Error(err.detail || "Pre-trained model watermarking failed");
  }
  return await res.json();
}

export function getDownloadUrl(relativePath: string): string {
  return `${API_BASE}${relativePath}`;
}

export async function verifyOwnership(data: {
  secret_key: string;
  secret_phrase: string;
  input_shape: number[];
  num_classes: number;
  target_class: number;
  num_triggers: number;
  pattern_type: string;
  endpoint_url?: string;
}): Promise<VerifyResponse> {
  const res = await fetch(`${API_BASE}/api/watermark/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Ownership verification failed");
  return await res.json();
}

export async function compareModels(formData: FormData): Promise<ModelComparisonResponse> {
  const res = await fetch(`${API_BASE}/api/watermark/compare-models`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Model comparison failed" }));
    throw new Error(err.detail || "Model comparison failed");
  }
  return await res.json();
}

export interface BenchmarkReport {
  dataset: string;
  model_architecture: string;
  baseline_model: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
  };
  watermarked_model: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    accuracy_delta: number;
    task_retention_rate: number;
  };
  watermark_verification: {
    num_triggers: number;
    trigger_matches: number;
    trigger_accuracy: number;
    p_value: number;
    verdict: string;
  };
  counterfeit_challenge: {
    false_alarm_matches: number;
    false_alarm_rate: number;
    verdict: string;
  };
  pruning_robustness: {
    [key: string]: number;
  };
}

export interface AttackTestResult {
  attack_type: string;
  attack_parameter: string;
  watermark_success_rate: number;
  trigger_matches: number;
  total_triggers: number;
  p_value: number;
  attack_repelled: boolean;
  status: string;
  details: string;
}

export async function fetchBenchmarkSummary(): Promise<BenchmarkReport | null> {
  try {
    const res = await fetch(`${API_BASE}/api/benchmarks/summary`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function runAttackTest(data: {
  attack_type: "counterfeit_key" | "pruning" | "quantization";
  model_id?: string;
  secret_key?: string;
  input_shape?: number[];
  num_classes?: number;
  target_class?: number;
  num_triggers?: number;
  prune_ratio?: number;
}): Promise<AttackTestResult> {
  const res = await fetch(`${API_BASE}/api/watermark/attack-test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Attack simulation failed" }));
    throw new Error(err.detail || "Attack simulation failed");
  }
  return await res.json();
}

// --- Live Pipeline Transparency Demo Endpoints ---

export interface DemoDatasetInfo {
  step: number;
  title: string;
  dataset_name: string;
  file_path: string;
  file_exists_on_disk: boolean;
  file_size_bytes: number;
  total_test_samples: number;
  image_shape: number[];
  channels: number;
  num_classes: number;
  class_labels: string[];
  samples: Array<{ index: number; label: number; name: string }>;
  transparency_note: string;
}

export interface DemoBaselineResult {
  step: number;
  title: string;
  model_file: string;
  file_exists_on_disk: boolean;
  file_size_bytes: number;
  architecture: string;
  total_parameters: number;
  weights_sha256: string;
  clean_accuracy: number;
  clean_accuracy_pct: string;
  test_images_evaluated: number;
  inference_latency_ms: number;
  sample_evaluations: Array<{ sample_idx: number; ground_truth: number; prediction: number; correct: boolean; confidence: number }>;
  status: string;
  note: string;
}

export interface DemoTriggerResult {
  step: number;
  title: string;
  key_cryptography: string;
  num_triggers: number;
  target_class: number;
  patch_location: string;
  patch_dimensions: string;
  random_guess_probability: number;
  probability_formula: string;
  triggers: Array<{ trigger_id: number; base_image_index: number; target_class: number; target_label: string; patch_location: string; patch_size: string }>;
  status: string;
  non_repudiation_note: string;
}

export interface DemoDistillationResult {
  step: number;
  title: string;
  distillation_method: string;
  total_epochs: number;
  duration_seconds: number;
  clean_accuracy_before: number;
  clean_accuracy_after: number;
  clean_task_retention: number;
  watermarked_model_file: string;
  file_exists_on_disk: boolean;
  file_size_bytes: number;
  weights_sha256: string;
  epoch_telemetry: Array<{ epoch: number; loss: number; distill_kl_loss: number; watermark_ce_loss: number; accuracy: number }>;
  status: string;
  note: string;
}

export interface DemoVerificationResult {
  step: number;
  title: string;
  genuine_test: {
    test_name: string;
    total_triggers: number;
    correct_matches: number;
    wsr: number;
    wsr_pct: string;
    p_value: number;
    p_value_formatted: string;
    verdict: string;
    status: string;
  };
  counterfeit_test: {
    test_name: string;
    total_triggers: number;
    correct_matches: number;
    far: number;
    far_pct: string;
    p_value: number;
    verdict: string;
    status: string;
  };
  pruning_test: {
    test_name: string;
    prune_ratio: number;
    total_triggers: number;
    surviving_matches: number;
    wsr: number;
    wsr_pct: string;
    operational_threshold: number;
    threshold_pct: string;
    p_value: number;
    verdict: string;
    status: string;
  };
  status: string;
  academic_summary: string;
}

export async function fetchDemoDataset(): Promise<DemoDatasetInfo> {
  const res = await fetch(`${API_BASE}/api/demo/inspect-dataset`, { cache: "no-store" });
  if (!res.ok) throw new Error("Dataset inspection failed");
  return await res.json();
}

export async function runDemoBaseline(): Promise<DemoBaselineResult> {
  const res = await fetch(`${API_BASE}/api/demo/evaluate-baseline`, { method: "POST" });
  if (!res.ok) throw new Error("Baseline evaluation failed");
  return await res.json();
}

export async function runDemoTriggers(): Promise<DemoTriggerResult> {
  const res = await fetch(`${API_BASE}/api/demo/generate-triggers`, { method: "POST" });
  if (!res.ok) throw new Error("Trigger generation failed");
  return await res.json();
}

export async function runDemoDistillation(): Promise<DemoDistillationResult> {
  const res = await fetch(`${API_BASE}/api/demo/run-distillation`, { method: "POST" });
  if (!res.ok) throw new Error("Live distillation failed");
  return await res.json();
}

export async function runDemoVerification(): Promise<DemoVerificationResult> {
  const res = await fetch(`${API_BASE}/api/demo/verify-step`, { method: "POST" });
  if (!res.ok) throw new Error("Verification step failed");
  return await res.json();
}

