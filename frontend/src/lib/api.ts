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
  watermark_accuracy: number;
  pruning_survival: number;
  quantization_survival: number;
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
