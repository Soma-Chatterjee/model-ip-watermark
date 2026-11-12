"""
Persistent PostgreSQL Repository for Model IP Protection SaaS (Supabase).
Handles saving/loading organizations, registered models, watermark keys,
authorship audits, and dispute verification logs.
Falls back gracefully if DATABASE_URL is not configured.
"""

import os
import json
import uuid
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


class DatabaseRepository:
    """Thread-safe / stateless PostgreSQL repository using Supabase Transaction Pooler."""

    @staticmethod
    def _get_conn():
        db_url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URL_DIRECT")
        if not db_url:
            return None
        return psycopg2.connect(db_url, connect_timeout=5)

    @classmethod
    def is_connected(cls) -> bool:
        try:
            conn = cls._get_conn()
            if not conn:
                return False
            conn.close()
            return True
        except Exception:
            return False

    @classmethod
    def ensure_organization(cls, slug: str, name: Optional[str] = None) -> Optional[str]:
        """Gets or creates an organization by slug and returns its UUID."""
        conn = cls._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM public.organizations WHERE slug = %s", (slug,))
                row = cur.fetchone()
                if row:
                    return str(row[0])
                org_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO public.organizations (id, name, slug) VALUES (%s, %s, %s)",
                    (org_id, name or slug, slug)
                )
                conn.commit()
                return org_id
        except Exception:
            return None
        finally:
            conn.close()

    @classmethod
    def save_registered_model(
        cls,
        model_id: str,
        org_slug: str,
        model_name: str,
        architecture: str,
        input_shape: List[int],
        num_classes: int,
        weights_sha256: str,
        status: str = "watermarked",
        key_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Persists a registered & watermarked model and its cryptographic key metadata."""
        conn = cls._get_conn()
        if not conn:
            return False
        try:
            org_id = cls.ensure_organization(org_slug)
            # Generate deterministic UUID from short model_id if not already UUID
            try:
                db_model_uuid = str(uuid.UUID(model_id))
            except ValueError:
                db_model_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, model_id))

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.registered_models
                    (id, organization_id, name, version, architecture, framework, input_shape, num_classes, weights_sha256, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, weights_sha256 = EXCLUDED.weights_sha256, updated_at = NOW()
                    """,
                    (
                        db_model_uuid,
                        org_id,
                        model_name,
                        model_id,
                        architecture,
                        "PyTorch",
                        json.dumps(input_shape),
                        num_classes,
                        weights_sha256,
                        status,
                    ),
                )

                if key_data:
                    cur.execute(
                        """
                        INSERT INTO public.watermark_keys
                        (id, model_id, owner_id, key_hash, encrypted_key, signature, num_triggers, target_class, trigger_seed)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid4()),
                            db_model_uuid,
                            org_slug,
                            key_data.get("key_hash", ""),
                            key_data.get("encrypted_key", "vault_managed"),
                            key_data.get("signature", ""),
                            key_data.get("num_triggers", 100),
                            key_data.get("target_class", 0),
                            key_data.get("trigger_seed", 0),
                        ),
                    )
                conn.commit()
            return True
        except Exception as e:
            print(f"[DB Error save_registered_model]: {e}")
            return False
        finally:
            conn.close()

    @classmethod
    def save_audit_log(
        cls,
        model_name: str,
        weights_sha256: str,
        dataset_hash: str,
        code_repo_url: Optional[str],
        approved: bool,
        confidence_score: int,
        summary: str
    ) -> bool:
        """Persists proof-of-authorship audit records."""
        conn = cls._get_conn()
        if not conn:
            return False
        try:
            db_model_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{model_name}_{weights_sha256}"))
            with conn.cursor() as cur:
                # Ensure parent model stub exists if audited before registration
                cur.execute(
                    """
                    INSERT INTO public.registered_models
                    (id, name, architecture, input_shape, num_classes, weights_sha256, status)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (db_model_uuid, model_name, "Custom", "[784]", 10, weights_sha256, "verified_original" if approved else "rejected"),
                )
                cur.execute(
                    """
                    INSERT INTO public.model_audits
                    (id, model_id, dataset_hash, code_repository_url, public_model_similarity_score, audit_decision, decision_reason, reviewed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        str(uuid.uuid4()),
                        db_model_uuid,
                        dataset_hash,
                        code_repo_url,
                        round(1.0 - (confidence_score / 100.0), 4),
                        "approved" if approved else "flagged_plagiarism",
                        summary,
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"[DB Error save_audit_log]: {e}")
            return False
        finally:
            conn.close()

    @classmethod
    def save_verification_log(
        cls,
        model_id: str,
        mode: str,
        endpoint_url: Optional[str],
        total_tested: int,
        correct_matches: int,
        accuracy: float,
        p_value: float,
        is_watermarked: bool,
        certificate_id: str
    ) -> bool:
        """Persists dispute verification query results."""
        conn = cls._get_conn()
        if not conn:
            return False
        try:
            try:
                db_model_uuid = str(uuid.UUID(model_id))
            except ValueError:
                db_model_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, model_id))

            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM public.registered_models WHERE id = %s", (db_model_uuid,))
                if not cur.fetchone():
                    return False
                cur.execute(
                    """
                    INSERT INTO public.verification_logs
                    (id, model_id, suspect_endpoint, verification_mode, total_triggers_tested, successful_triggers, accuracy, p_value, is_watermarked, certificate_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        db_model_uuid,
                        endpoint_url,
                        mode,
                        total_tested,
                        correct_matches,
                        round(accuracy, 4),
                        min(1.0, max(0.0, float(p_value))),
                        is_watermarked,
                        certificate_id,
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"[DB Error save_verification_log]: {e}")
            return False
        finally:
            conn.close()

    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        """Fetches all persistent watermarked models from Supabase."""
        conn = cls._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT m.version AS id, m.name AS model_name, COALESCE(o.slug, 'default') AS owner_id,
                           m.architecture, m.weights_sha256, m.status, m.created_at,
                           k.target_class
                    FROM public.registered_models m
                    LEFT JOIN public.organizations o ON m.organization_id = o.id
                    LEFT JOIN public.watermark_keys k ON m.id = k.model_id
                    WHERE m.status = 'watermarked'
                    ORDER BY m.created_at DESC
                    LIMIT 50
                    """
                )
                rows = cur.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "model_name": r["model_name"],
                        "owner_id": r["owner_id"],
                        "architecture": r["architecture"],
                        "target_class": r["target_class"] if r["target_class"] is not None else 0,
                        "watermark_accuracy": 1.0,
                        "pruning_survival": 1.0,
                        "quantization_survival": 1.0,
                        "weights_sha256": r["weights_sha256"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else "",
                    })
                return results
        except Exception as e:
            print(f"[DB Error list_models]: {e}")
            return []
        finally:
            conn.close()
