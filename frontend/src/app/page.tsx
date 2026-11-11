"use client";

import React, { useState, useEffect } from "react";
import {
  ShieldCheck,
  KeyRound,
  FileCheck2,
  Cpu,
  Fingerprint,
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Lock,
  Layers,
  Sparkles,
  Terminal,
  ExternalLink,
  Copy,
  Check,
} from "lucide-react";
import {
  checkBackendHealth,
  fetchProtectedModels,
  generateWatermarkKey,
  verifyAuthorship,
  embedWatermark,
  verifyOwnership,
  KeyGenResponse,
  AuditResponse,
  EmbedResponse,
  VerifyResponse,
  ProtectedModelInfo,
} from "@/lib/api";
import confetti from "canvas-confetti";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "audit" | "embed" | "verify">("dashboard");
  const [backendStatus, setBackendStatus] = useState<{ status: string; uptime_seconds: number } | null>(null);
  const [activeModels, setActiveModels] = useState<ProtectedModelInfo[]>([]);
  const [copiedKey, setCopiedKey] = useState(false);

  // Form states (empty by default for user entry)
  const [modelName, setModelName] = useState("");
  const [ownerId, setOwnerId] = useState("");
  const [keyData, setKeyData] = useState<KeyGenResponse | null>(null);

  // Audit states
  const [weightsHash, setWeightsHash] = useState("");
  const [datasetHash, setDatasetHash] = useState("");
  const [trainingLogs, setTrainingLogs] = useState("");
  const [repoUrl, setRepoUrl] = useState("");
  const [auditResult, setAuditResult] = useState<AuditResponse | null>(null);
  const [isAuditing, setIsAuditing] = useState(false);

  // Embedding states
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [embedResult, setEmbedResult] = useState<EmbedResponse | null>(null);

  // Verification states
  const [verifyEndpoint, setVerifyEndpoint] = useState("");
  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null);

  const refreshEngineState = () => {
    checkBackendHealth().then(setBackendStatus);
    fetchProtectedModels().then((data) => setActiveModels(data.models));
  };

  // Check backend health & real active models periodically
  useEffect(() => {
    refreshEngineState();
    const interval = setInterval(refreshEngineState, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleGenerateKey = async () => {
    if (!modelName.trim() || !ownerId.trim()) {
      alert("Please enter both a Target Model Name and Owner Identifier.");
      return;
    }
    try {
      const res = await generateWatermarkKey({
        model_name: modelName.trim(),
        owner_id: ownerId.trim(),
        input_shape: [784],
        num_triggers: 100,
        target_class: 7,
        pattern_type: "pattern",
      });
      setKeyData(res);
      confetti({ particleCount: 40, spread: 60, origin: { y: 0.8 } });
    } catch (err: any) {
      alert("Error generating key: " + err.message);
    }
  };

  const handleRunAudit = async () => {
    setIsAuditing(true);
    try {
      const res = await verifyAuthorship({
        model_name: modelName,
        weights_sha256: weightsHash,
        dataset_hash: datasetHash,
        training_logs_raw: trainingLogs,
        code_repo_url: repoUrl,
      });
      setAuditResult(res);
      if (res.approved) {
        confetti({ particleCount: 50, spread: 70, origin: { y: 0.7 } });
      }
    } catch (err: any) {
      alert("Audit failed: " + err.message);
    } finally {
      setIsAuditing(false);
    }
  };

  const handleEmbedWatermark = async () => {
    if (!keyData) {
      alert("Please generate or provide a Cryptographic Key first (Step 1).");
      return;
    }
    setIsEmbedding(true);
    try {
      const res = await embedWatermark({
        model_name: modelName,
        owner_id: ownerId,
        secret_key: keyData.secret_key,
        architecture: "MLP",
        input_shape: [784],
        num_classes: 10,
        target_class: 7,
        epochs: 4,
        lambda_wm: 0.1,
      });
      setEmbedResult(res);
      refreshEngineState();
      confetti({ particleCount: 80, spread: 90, origin: { y: 0.6 } });
    } catch (err: any) {
      alert("Watermarking failed: " + err.message);
    } finally {
      setIsEmbedding(false);
    }
  };

  const handleRunVerification = async () => {
    if (!keyData) {
      alert("Please generate a key or paste your secret watermark key first.");
      return;
    }
    setIsVerifying(true);
    try {
      const res = await verifyOwnership({
        secret_key: keyData.secret_key,
        input_shape: [784],
        num_classes: 10,
        target_class: 7,
        num_triggers: 100,
        pattern_type: "pattern",
        endpoint_url: verifyEndpoint ? verifyEndpoint : undefined,
      });
      setVerifyResult(res);
      if (res.is_watermarked) {
        confetti({ particleCount: 100, spread: 100, origin: { y: 0.5 } });
      }
    } catch (err: any) {
      alert("Verification query failed: " + err.message);
    } finally {
      setIsVerifying(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2000);
  };

  return (
    <div className="min-h-screen bg-[#090d14] text-slate-100 flex flex-col selection:bg-[#ff453a]/30 selection:text-white">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-[#0d121c]/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#ff453a] to-[#ff2d55] flex items-center justify-center shadow-lg shadow-[#ff453a]/20">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-semibold tracking-tight text-white flex items-center gap-1.5">
                Antigravity <span className="text-[#ff453a] font-mono text-xs px-1.5 py-0.5 rounded bg-[#ff453a]/10 border border-[#ff453a]/30">IP-PASS</span>
              </span>
              <p className="text-[11px] text-slate-400 font-mono">Neural Model Watermarking & Dispute Authority</p>
            </div>
          </div>

          {/* Render Heartbeat Status Indicator */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-full bg-slate-900/80 border border-slate-800">
              <span className={`w-2 h-2 rounded-full ${backendStatus ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
              <span className="text-slate-300">
                {backendStatus ? `Engine Active (${backendStatus.uptime_seconds}s)` : "Connecting to Engine..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Hero / Header Command bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-8 pb-6 w-full">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800/60">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
              Model Protection Command Center
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Deterministic cryptographic trigger-set injection, zero-knowledge verification & anti-plagiarism origin audits.
            </p>
          </div>

          {/* Raycast-style Segmented Controller */}
          <div className="flex p-1 rounded-xl bg-[#111723] border border-slate-800/80 self-start md:self-auto text-xs font-medium">
            <button
              onClick={() => setActiveTab("dashboard")}
              className={`px-3.5 py-2 rounded-lg transition-all flex items-center gap-2 ${
                activeTab === "dashboard"
                  ? "bg-[#ff453a] text-white shadow-md shadow-[#ff453a]/25"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              Overview
            </button>
            <button
              onClick={() => setActiveTab("audit")}
              className={`px-3.5 py-2 rounded-lg transition-all flex items-center gap-2 ${
                activeTab === "audit"
                  ? "bg-[#ff453a] text-white shadow-md shadow-[#ff453a]/25"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <FileCheck2 className="w-3.5 h-3.5" />
              Proof of Origin
            </button>
            <button
              onClick={() => setActiveTab("embed")}
              className={`px-3.5 py-2 rounded-lg transition-all flex items-center gap-2 ${
                activeTab === "embed"
                  ? "bg-[#ff453a] text-white shadow-md shadow-[#ff453a]/25"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              Watermark Injection
            </button>
            <button
              onClick={() => setActiveTab("verify")}
              className={`px-3.5 py-2 rounded-lg transition-all flex items-center gap-2 ${
                activeTab === "verify"
                  ? "bg-[#ff453a] text-white shadow-md shadow-[#ff453a]/25"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Fingerprint className="w-3.5 h-3.5" />
              Verify Dispute
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 pb-16 w-full flex-1">
        {/* TAB 1: OVERVIEW & KEY VAULT */}
        {activeTab === "dashboard" && (
          <div className="space-y-6">
            {/* Top Stat Banners */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="raycast-card p-4 rounded-xl">
                <span className="text-xs text-slate-400 font-mono">Protected Models</span>
                <p className="text-2xl font-bold mt-1 text-white">{activeModels.length} Active</p>
                <span className="text-[11px] text-emerald-400 flex items-center gap-1 mt-1 font-mono">
                  <CheckCircle2 className="w-3 h-3" /> {activeModels.length > 0 ? "Live in Engine Registry" : "Ready for Registration"}
                </span>
              </div>
              <div className="raycast-card p-4 rounded-xl">
                <span className="text-xs text-slate-400 font-mono">Key Cryptography</span>
                <p className="text-2xl font-bold mt-1 text-white">Fernet 256-bit</p>
                <span className="text-[11px] text-cyan-400 flex items-center gap-1 mt-1 font-mono">
                  Deterministic Seed Derivation
                </span>
              </div>
              <div className="raycast-card p-4 rounded-xl">
                <span className="text-xs text-slate-400 font-mono">Verification Math</span>
                <p className="text-2xl font-bold mt-1 text-white">p &lt; 10⁻²⁰</p>
                <span className="text-[11px] text-slate-400 flex items-center gap-1 mt-1 font-mono">
                  Binomial Exact Confidence
                </span>
              </div>
              <div className="raycast-card p-4 rounded-xl">
                <span className="text-xs text-slate-400 font-mono">Authorship Gate</span>
                <p className="text-2xl font-bold mt-1 text-white">Anti-Plagiarism</p>
                <span className="text-[11px] text-emerald-400 flex items-center gap-1 mt-1 font-mono">
                  Public Model Collision Check
                </span>
              </div>
            </div>

            {/* Step 1: Cryptographic Key Generator */}
            <div className="raycast-panel p-6 rounded-2xl border border-slate-800">
              <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-[#ff453a]/10 text-[#ff453a] border border-[#ff453a]/20">
                    <KeyRound className="w-5 h-5" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">1. Master Key & Trigger Set Generator</h2>
                    <p className="text-xs text-slate-400">
                      Derives an unforgeable trigger set deterministically from a 256-bit symmetric key.
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleGenerateKey}
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-[#ff453a] to-[#ff2d55] text-white text-xs font-semibold hover:opacity-90 transition shadow-lg shadow-[#ff453a]/25 flex items-center gap-2"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  Generate New Master Key
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div>
                  <label className="text-xs font-mono text-slate-400">Target Model Name</label>
                  <input
                    type="text"
                    placeholder="e.g. ResNet18-MedicalClassifier"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="w-full mt-1.5 px-3 py-2 rounded-lg raycast-input text-sm text-slate-200 font-mono"
                  />
                </div>
                <div>
                  <label className="text-xs font-mono text-slate-400">Owner Identifier / Organization</label>
                  <input
                    type="text"
                    placeholder="e.g. org_deephealth_labs"
                    value={ownerId}
                    onChange={(e) => setOwnerId(e.target.value)}
                    className="w-full mt-1.5 px-3 py-2 rounded-lg raycast-input text-sm text-slate-200 font-mono"
                  />
                </div>
              </div>

              {keyData && (
                <div className="mt-5 p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-3 font-mono text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-emerald-400 font-semibold flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4" /> Cryptographic Key Generated & Verified
                    </span>
                    <button
                      onClick={() => copyToClipboard(keyData.secret_key)}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1"
                    >
                      {copiedKey ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                      {copiedKey ? "Copied" : "Copy Key"}
                    </button>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-slate-300">
                    <div>
                      <span className="text-slate-500 block">Secret Fernet Key:</span>
                      <p className="truncate text-cyan-300 mt-0.5">{keyData.secret_key}</p>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Key SHA-256 Hash:</span>
                      <p className="truncate text-slate-400 mt-0.5">{keyData.key_hash}</p>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Trigger Deterministic Seed:</span>
                      <p className="text-slate-200 mt-0.5">{keyData.trigger_seed}</p>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Target Watermark Class:</span>
                      <p className="text-[#ff453a] font-bold mt-0.5">Class {keyData.target_class}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Live Active Models Registry */}
            <div className="raycast-panel p-6 rounded-2xl border border-slate-800">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center justify-between">
                <span>Live Protected Models Registry</span>
                <span className="text-xs font-mono text-slate-400">{activeModels.length} Registered</span>
              </h3>
              {activeModels.length === 0 ? (
                <div className="text-xs font-mono text-slate-500 p-6 rounded-xl bg-slate-950/50 border border-slate-800/60 text-center">
                  No models are currently protected in this engine instance. Generate a master key above and inject a watermark to register your first model.
                </div>
              ) : (
                <div className="space-y-2">
                  {activeModels.map((m) => (
                    <div
                      key={m.id}
                      className="flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs font-mono gap-2"
                    >
                      <div>
                        <span className="text-white font-semibold">{m.model_name}</span>
                        <span className="text-slate-500 ml-2">({m.owner_id})</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-emerald-400">WM Acc: {(m.watermark_accuracy * 100).toFixed(0)}%</span>
                        <span className="text-cyan-400">Prune: {(m.pruning_survival * 100).toFixed(0)}%</span>
                        <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300">ID: {m.id}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: PROOF OF ORIGIN & ANTI-PLAGIARISM AUDIT */}
        {activeTab === "audit" && (
          <div className="raycast-panel p-6 rounded-2xl border border-slate-800 space-y-6">
            <div className="flex items-center space-x-3 pb-4 border-b border-slate-800/80">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <FileCheck2 className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Proof of Authorship & Origin Verification</h2>
                <p className="text-xs text-slate-400">
                  Manual & automated lineage checks: confirms model was genuinely trained from scratch and not copied from HuggingFace.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-mono text-slate-400">Model Weights SHA-256 Fingerprint</label>
                <input
                  type="text"
                  placeholder="64-character SHA-256 hex digest of model weights"
                  value={weightsHash}
                  onChange={(e) => setWeightsHash(e.target.value)}
                  className="w-full mt-1.5 px-3 py-2 rounded-lg raycast-input text-sm text-slate-200 font-mono"
                />
              </div>
              <div>
                <label className="text-xs font-mono text-slate-400">Training Dataset SHA-256 Hash</label>
                <input
                  type="text"
                  placeholder="64-character SHA-256 hex digest of training dataset"
                  value={datasetHash}
                  onChange={(e) => setDatasetHash(e.target.value)}
                  className="w-full mt-1.5 px-3 py-2 rounded-lg raycast-input text-sm text-slate-200 font-mono"
                />
              </div>
              <div className="md:col-span-2">
                <label className="text-xs font-mono text-slate-400">Training Loss Log / WandB Curve Export</label>
                <textarea
                  rows={3}
                  placeholder="Paste training logs (e.g. Epoch 1: main_loss = 2.45 ...)"
                  value={trainingLogs}
                  onChange={(e) => setTrainingLogs(e.target.value)}
                  className="w-full mt-1.5 px-3 py-2 rounded-lg raycast-input text-xs text-slate-200 font-mono"
                />
              </div>
              <div className="md:col-span-2">
                <label className="text-xs font-mono text-slate-400">Code Repository / Training Artifact URL</label>
                <input
                  type="text"
                  placeholder="https://github.com/your-org/your-model-repo"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  className="w-full mt-1.5 px-3 py-2 rounded-lg raycast-input text-sm text-slate-200 font-mono"
                />
              </div>
            </div>

            <button
              onClick={handleRunAudit}
              disabled={isAuditing}
              className="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold transition flex items-center gap-2 shadow-lg shadow-cyan-600/20 disabled:opacity-50"
            >
              {isAuditing ? "Auditing Documentation..." : "Run Authorship Audit"}
            </button>

            {auditResult && (
              <div
                className={`p-4 rounded-xl border ${
                  auditResult.approved
                    ? "bg-emerald-950/30 border-emerald-500/40 text-emerald-300"
                    : "bg-red-950/30 border-red-500/40 text-red-300"
                } text-xs font-mono space-y-2`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-sm flex items-center gap-1.5">
                    {auditResult.approved ? <CheckCircle2 className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                    Audit Decision: {auditResult.status.toUpperCase()}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-black/40 border border-slate-700">
                    Confidence Score: {auditResult.confidence_score}/100
                  </span>
                </div>
                <p className="text-slate-300">{auditResult.summary}</p>
                {auditResult.issues.length > 0 && (
                  <ul className="list-disc pl-5 text-red-400 space-y-1">
                    {auditResult.issues.map((iss, idx) => (
                      <li key={idx}>{iss}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: WATERMARK INJECTION TRAINING */}
        {activeTab === "embed" && (
          <div className="raycast-panel p-6 rounded-2xl border border-slate-800 space-y-6">
            <div className="flex items-center space-x-3 pb-4 border-b border-slate-800/80">
              <div className="p-2 rounded-lg bg-[#ff453a]/10 text-[#ff453a] border border-[#ff453a]/20">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Watermark Injection & Attack Resilience</h2>
                <p className="text-xs text-slate-400">
                  Injects triggers into parameters via joint loss: L_total = L_main + λ·L_wm, then tests against 30% pruning and 8-bit quantization.
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/50 border border-slate-800/80 text-xs font-mono text-slate-300 flex items-center justify-between">
              <div>
                <span className="text-slate-500 block">Selected Model:</span>
                <span className="text-white font-bold">{modelName}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Active Fernet Key:</span>
                <span className="text-cyan-400">{keyData ? keyData.key_hash.slice(0, 16) + "..." : "No Key Generated"}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Loss Weight (λ):</span>
                <span className="text-[#ff453a] font-bold">0.1 (Balanced)</span>
              </div>
            </div>

            <button
              onClick={handleEmbedWatermark}
              disabled={isEmbedding || !keyData}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#ff453a] to-[#ff2d55] text-white text-xs font-semibold hover:opacity-90 transition shadow-lg shadow-[#ff453a]/25 disabled:opacity-50 flex items-center gap-2"
            >
              {isEmbedding ? "Embedding Watermark & Simulating Attacks..." : "Embed Watermark Now"}
            </button>

            {embedResult && (
              <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-4 font-mono text-xs">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <span className="text-emerald-400 font-bold flex items-center gap-1.5 text-sm">
                    <CheckCircle2 className="w-4 h-4" /> Watermark Successfully Embedded
                  </span>
                  <span className="text-slate-400">Trigger Accuracy: {(embedResult.final_watermark_accuracy * 100).toFixed(1)}%</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block">Final Watermark Accuracy</span>
                    <p className="text-lg font-bold text-emerald-400 mt-1">{(embedResult.final_watermark_accuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block">Survives 30% Weight Pruning</span>
                    <p className="text-lg font-bold text-cyan-400 mt-1">{(embedResult.pruning_survival_accuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block">Survives 8-bit Quantization</span>
                    <p className="text-lg font-bold text-cyan-400 mt-1">{(embedResult.quantization_survival_accuracy * 100).toFixed(1)}%</p>
                  </div>
                </div>

                <div className="pt-2">
                  <span className="text-slate-500 block mb-1">Epoch Trajectory:</span>
                  <div className="space-y-1">
                    {embedResult.training_history.map((h, i) => (
                      <div key={i} className="flex justify-between text-slate-400 text-[11px] bg-slate-900/50 px-2 py-1 rounded">
                        <span>Epoch {h.epoch}</span>
                        <span>Main Loss: {h.main_loss}</span>
                        <span>WM Loss: {h.watermark_loss}</span>
                        <span className="text-emerald-400 font-bold">WM Acc: {(h.watermark_accuracy * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: DISPUTE & VERIFICATION ENGINE */}
        {activeTab === "verify" && (
          <div className="raycast-panel p-6 rounded-2xl border border-slate-800 space-y-6">
            <div className="flex items-center space-x-3 pb-4 border-b border-slate-800/80">
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Fingerprint className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Dispute Resolution & Ownership Verification</h2>
                <p className="text-xs text-slate-400">
                  Challenges a suspect model endpoint or local checkpoint with 100 secret triggers to compute binomial p-value confidence.
                </p>
              </div>
            </div>

            <div>
              <label className="text-xs font-mono text-slate-400">Suspect Model API Endpoint (Leave empty to test registered model checkpoint)</label>
              <input
                type="text"
                placeholder="https://api.competitor-ai.com/v1/predict"
                value={verifyEndpoint}
                onChange={(e) => setVerifyEndpoint(e.target.value)}
                className="w-full mt-1.5 px-3 py-2 rounded-lg raycast-input text-sm text-slate-200 font-mono"
              />
            </div>

            <button
              onClick={handleRunVerification}
              disabled={isVerifying || !keyData}
              className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition flex items-center gap-2 shadow-lg shadow-emerald-600/25 disabled:opacity-50"
            >
              {isVerifying ? "Challenging Suspect Model..." : "Launch Cryptographic Verification"}
            </button>

            {verifyResult && (
              <div
                className={`p-5 rounded-xl border ${
                  verifyResult.is_watermarked
                    ? "bg-emerald-950/40 border-emerald-500/50"
                    : "bg-red-950/40 border-red-500/50"
                } text-xs font-mono space-y-4`}
              >
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <span
                    className={`font-bold text-sm flex items-center gap-2 ${
                      verifyResult.is_watermarked ? "text-emerald-400" : "text-red-400"
                    }`}
                  >
                    {verifyResult.is_watermarked ? <CheckCircle2 className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                    VERDICT: {verifyResult.verdict}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-black/50 border border-slate-700 text-slate-300">
                    Cert: {verifyResult.certificate_id}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-slate-200">
                  <div>
                    <span className="text-slate-500 block">Trigger Matches:</span>
                    <span className="text-base font-bold text-white">
                      {verifyResult.correct_matches} / {verifyResult.total_tested}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Trigger Accuracy:</span>
                    <span className="text-base font-bold text-emerald-400">
                      {(verifyResult.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Binomial p-value:</span>
                    <span className="text-base font-bold text-cyan-400">
                      {verifyResult.p_value < 1e-6 ? verifyResult.p_value.toExponential(4) : verifyResult.p_value.toFixed(6)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Statistical Confidence:</span>
                    <span className="text-base font-bold text-white">
                      {verifyResult.confidence_percent.toFixed(4)}%
                    </span>
                  </div>
                </div>

                {verifyResult.is_watermarked && (
                  <p className="text-slate-300 pt-2 border-t border-slate-800/80">
                    Mathematical Proof: The chance of an unwatermarked model producing this trigger response distribution purely by coincidence is{" "}
                    <span className="text-cyan-300 font-bold">1 in {Math.round(1 / Math.max(1e-15, verifyResult.p_value)).toLocaleString()}</span>. This constitutes legally admissible ownership evidence.
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 text-center text-xs text-slate-500 font-mono">
        Antigravity Model IP Authority &copy; 2026. Cryptographically Verified & Backdoor-Resilient.
      </footer>
    </div>
  );
}
