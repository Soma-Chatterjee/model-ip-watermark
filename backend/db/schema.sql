-- Supabase Schema for Model IP Protection & Watermarking SaaS

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Organizations / Teams Table
CREATE TABLE IF NOT EXISTS public.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Registered Models Table
CREATE TABLE IF NOT EXISTS public.registered_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT 'v1.0.0',
    architecture TEXT NOT NULL, -- e.g., 'ResNet-18', 'CNN-Classifier', 'MLP', 'Custom-DNN'
    framework TEXT NOT NULL DEFAULT 'PyTorch',
    input_shape JSONB NOT NULL, -- e.g., [1, 28, 28] or [3, 32, 32]
    num_classes INT NOT NULL,
    weights_sha256 TEXT NOT NULL, -- SHA-256 fingerprint of the model weights
    status TEXT NOT NULL DEFAULT 'draft', -- 'draft', 'pending_audit', 'verified_original', 'watermarked', 'rejected'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Proof-of-Authorship & Documentation Audit Table
CREATE TABLE IF NOT EXISTS public.model_audits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES public.registered_models(id) ON DELETE CASCADE,
    dataset_hash TEXT NOT NULL,
    training_logs_url TEXT,
    code_repository_url TEXT,
    public_model_similarity_score NUMERIC(5, 4) DEFAULT 0.0000, -- 0.0 to 1.0 (e.g. CKA or cosine similarity vs known public baselines)
    audit_decision TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'approved', 'flagged_plagiarism'
    decision_reason TEXT,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Cryptographic Watermark Keys Table
CREATE TABLE IF NOT EXISTS public.watermark_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES public.registered_models(id) ON DELETE CASCADE,
    owner_id TEXT NOT NULL,
    key_hash TEXT NOT NULL, -- SHA-256 of the Fernet key (never store raw key unencrypted)
    encrypted_key TEXT NOT NULL, -- Fernet key encrypted or vault-secured
    signature TEXT NOT NULL, -- Cryptographic signature binding model + owner + timestamp
    num_triggers INT NOT NULL DEFAULT 100,
    target_class INT NOT NULL,
    trigger_seed BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Verification & Dispute Resolution Logs
CREATE TABLE IF NOT EXISTS public.verification_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES public.registered_models(id) ON DELETE CASCADE,
    suspect_endpoint TEXT,
    verification_mode TEXT NOT NULL, -- 'black_box_api', 'white_box_weights'
    total_triggers_tested INT NOT NULL,
    successful_triggers INT NOT NULL,
    accuracy NUMERIC(5, 4) NOT NULL, -- e.g. 0.9800
    p_value NUMERIC(10, 8), -- binomial probability of random coincidence
    is_watermarked BOOLEAN NOT NULL,
    certificate_id TEXT UNIQUE,
    report_details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_registered_models_weights_sha256 ON public.registered_models(weights_sha256);
CREATE INDEX IF NOT EXISTS idx_watermark_keys_model_id ON public.watermark_keys(model_id);
CREATE INDEX IF NOT EXISTS idx_verification_logs_model_id ON public.verification_logs(model_id);
