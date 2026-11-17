-- =====================================================================
-- DeepMark / Supabase Security Fix: Enable Row-Level Security (RLS)
-- Fixes Supabase Critical Security Advisory: 'rls_disabled_in_public'
-- Project: wqgpufjmyxqaqwejuskn
-- =====================================================================

-- 1. Enable Row-Level Security (RLS) on all 5 public tables
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.registered_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.watermark_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_audits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.verification_logs ENABLE ROW LEVEL SECURITY;

-- 2. Drop any legacy/conflicting policies if they exist
DROP POLICY IF EXISTS "Allow public read access on organizations" ON public.organizations;
DROP POLICY IF EXISTS "Allow public read access on registered_models" ON public.registered_models;
DROP POLICY IF EXISTS "Allow public read access on verification_logs" ON public.verification_logs;
DROP POLICY IF EXISTS "Allow service role full access on organizations" ON public.organizations;
DROP POLICY IF EXISTS "Allow service role full access on registered_models" ON public.registered_models;
DROP POLICY IF EXISTS "Allow service role full access on watermark_keys" ON public.watermark_keys;
DROP POLICY IF EXISTS "Allow service role full access on model_audits" ON public.model_audits;
DROP POLICY IF EXISTS "Allow service role full access on verification_logs" ON public.verification_logs;

-- 3. Define Public Read Policies (Allows your Next.js frontend & public visitors to view verified models)
CREATE POLICY "Allow public read access on organizations"
    ON public.organizations
    FOR SELECT
    USING (true);

CREATE POLICY "Allow public read access on registered_models"
    ON public.registered_models
    FOR SELECT
    USING (status = 'watermarked');

CREATE POLICY "Allow public read access on verification_logs"
    ON public.verification_logs
    FOR SELECT
    USING (true);

-- 4. Define Restricted Policies (Protects secret keys, audits, and database modifications)
-- Only your authenticated backend (FastAPI / service_role) can insert/update/delete.
-- Anonymous attackers using the public anon key CANNOT tamper with or delete your data!

CREATE POLICY "Allow service role full access on organizations"
    ON public.organizations
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role full access on registered_models"
    ON public.registered_models
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role full access on watermark_keys"
    ON public.watermark_keys
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role full access on model_audits"
    ON public.model_audits
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role full access on verification_logs"
    ON public.verification_logs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
