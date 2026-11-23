"""BDWebs FastAPI Zero-LLM Semantic Engine (ONNX multilingual-e5-small + optional SetFit)."""

import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from normalizer import (
    enrich_query_for_embedding,
    extract_domain_tokens,
    is_gibberish_or_noise,
    is_semantic_eligible_rule,
    normalize_text,
    passes_rule_guards,
)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
TOPIC_BONUS = 0.03
_URL_RE = re.compile(r"https?://\S+", re.I)


def load_embedder(model_name: str) -> Any:
    """Load FastEmbed TextEmbedding, registering intfloat/multilingual-e5-small if not built-in."""
    from fastembed import TextEmbedding

    supported = {str(m.get("model", "")).lower() for m in TextEmbedding.list_supported_models()}
    if model_name.lower() == DEFAULT_MODEL.lower() and model_name.lower() not in supported:
        from fastembed.common.model_description import ModelSource, PoolingType

        TextEmbedding.add_custom_model(
            model=DEFAULT_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=DEFAULT_MODEL),
            dim=384,
            model_file="onnx/model.onnx",
            size_in_gb=0.47,
        )
    return TextEmbedding(model_name=model_name)


def _parse_keywords(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []


def re_clean_snippet(text: str) -> str:
    """Strip URLs and Markdown formatting symbols for clean semantic passage embeddings."""
    cleaned = _URL_RE.sub(" ", str(text or "").replace("\\\n", " "))
    cleaned = re.sub(r"[*`\\#<>]+", " ", cleaned)
    return " ".join(cleaned.split())


def build_rule_passage(rule: dict, use_e5_prefix: bool = True) -> str:
    """Format a KB rule cleanly into a natural passage string without template boilerplate."""
    title = re_clean_snippet(str(rule.get("title") or rule.get("rule_key") or ""))
    kws = _parse_keywords(rule.get("keywords"))
    dedup_kws: List[str] = []
    seen_kw: Set[str] = set()
    for kw in kws:
        cleaned_kw = re_clean_snippet(kw)
        low = cleaned_kw.lower()
        if cleaned_kw and low not in seen_kw and low != title.lower():
            seen_kw.add(low)
            dedup_kws.append(cleaned_kw)
    kw_str = ", ".join(dedup_kws)[:140].strip(", ")
    resp = re_clean_snippet(str(rule.get("response") or ""))[:260]
    parts = [p for p in (title, kw_str, resp) if p]
    body = ". ".join(parts)
    return f"passage: {body}" if use_e5_prefix else body


class SemanticIndex:
    """Thread-safe in-memory vector index backed by numpy and optional SetFit classifier."""

    def __init__(self, data_dir: str, embedder: Optional[Any] = None, use_e5_prefix: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.use_e5_prefix = use_e5_prefix
        self.setfit_model: Optional[Any] = None
        self.rules: List[dict] = []
        self.rule_tokens: List[Set[str]] = []
        self.vectors: Optional[np.ndarray] = None
        self._load_persisted()

    def _index_paths(self):
        return self.data_dir / "index.npz", self.data_dir / "rules.json"

    def _build_rule_tokens(self, rules: List[dict]) -> List[Set[str]]:
        out: List[Set[str]] = []
        for rule in rules:
            title = str(rule.get("title") or "")
            kws = " ".join(_parse_keywords(rule.get("keywords")))
            out.append(extract_domain_tokens(f"{title} {kws}"))
        return out

    def _load_persisted(self) -> None:
        idx_path, rules_path = self._index_paths()
        if idx_path.exists() and rules_path.exists():
            try:
                loaded_rules = json.loads(rules_path.read_text(encoding="utf-8"))
                npz = np.load(idx_path)
                vecs = npz["vectors"].astype(np.float32)
                if len(loaded_rules) == vecs.shape[0]:
                    self.rules = loaded_rules
                    self.rule_tokens = self._build_rule_tokens(loaded_rules)
                    self.vectors = vecs
            except Exception:
                self.rules = []
                self.rule_tokens = []
                self.vectors = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("Embedding model not loaded")
        raw = list(self.embedder.embed(texts))
        mat = np.asarray(raw, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    def reindex(self, raw_rules: List[dict]) -> Dict[str, Any]:
        started = time.perf_counter()
        eligible: List[dict] = []
        passages: List[str] = []
        skipped = 0
        for r in raw_rules:
            if not is_semantic_eligible_rule(r):
                skipped += 1
                continue
            eligible.append(r)
            passages.append(build_rule_passage(r, use_e5_prefix=self.use_e5_prefix))

        if not eligible:
            self.rules = []
            self.rule_tokens = []
            self.vectors = None
            return {"ok": True, "indexed_count": 0, "skipped_count": skipped, "elapsed_ms": 0.0}

        vecs = self._encode(passages)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        idx_path, rules_path = self._index_paths()
        tmp_idx = self.data_dir / "index.tmp.npz"
        tmp_rules = self.data_dir / "rules.tmp.json"
        np.savez_compressed(tmp_idx, vectors=vecs)
        tmp_rules.write_text(json.dumps(eligible, ensure_ascii=False), encoding="utf-8")
        tmp_idx.replace(idx_path)
        tmp_rules.replace(rules_path)

        self.rules = eligible
        self.rule_tokens = self._build_rule_tokens(eligible)
        self.vectors = vecs
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        return {"ok": True, "indexed_count": len(eligible), "skipped_count": skipped, "elapsed_ms": elapsed}

    def _setfit_boosts(self, query: str) -> Dict[str, float]:
        if self.setfit_model is None:
            return {}
        try:
            probs = self.setfit_model.predict_proba([normalize_text(query)])[0]
            labels = list(getattr(self.setfit_model, "labels", []) or [])
            return {str(lbl): float(p) for lbl, p in zip(labels, probs)}
        except Exception:
            return {}

    def match(
        self,
        query: str,
        *,
        mode: str = "shadow",
        active_topic: str = "",
        top_k: int = 3,
        direct_threshold: float = 0.84,
        suggest_threshold: float = 0.76,
        handoff_threshold: float = 0.88,
        min_z_score: float = 2.0,
        min_suggest_z_score: float = 1.4,
        min_margin: float = 0.045,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if mode == "off":
            return {"ok": True, "mode": "off", "decision": "bypass", "best_match": None, "suggestions": [], "latency_ms": 0.0}
        if not query.strip() or self.vectors is None or not self.rules:
            return {"ok": True, "mode": mode, "decision": "miss", "best_match": None, "suggestions": [], "latency_ms": 0.0}
        if is_gibberish_or_noise(query):
            latency = round((time.perf_counter() - started) * 1000, 1)
            return {"ok": True, "mode": mode, "decision": "miss", "best_match": None, "suggestions": [], "top_candidates": [], "latency_ms": latency}

        q_vec = self._encode([enrich_query_for_embedding(query, active_topic, use_e5_prefix=self.use_e5_prefix)])[0]
        sims = self.vectors @ q_vec
        med = float(np.median(sims))
        mean = float(np.mean(sims))
        std = max(float(np.std(sims)), 1e-4)

        sf_boosts = self._setfit_boosts(query)
        topic_token = active_topic.replace("_", " ").lower() if active_topic else ""
        q_domain_tokens = extract_domain_tokens(query)

        scored: List[Dict[str, Any]] = []
        for idx, sim in enumerate(sims.tolist()):
            rule = self.rules[idx]
            if not passes_rule_guards(rule, query):
                continue
            rk = str(rule.get("rule_key") or "")
            title = str(rule.get("title") or rk)
            raw_cos = float(sim)
            score = raw_cos
            if topic_token and (topic_token in title.lower() or topic_token.replace(" ", "_") in rk.lower()):
                score += TOPIC_BONUS
            r_tokens = self.rule_tokens[idx] if idx < len(self.rule_tokens) else set()
            lex_hits = len(q_domain_tokens & r_tokens) if q_domain_tokens and r_tokens else 0
            if lex_hits > 0:
                score += min(0.06, 0.025 * lex_hits)
            if rk in sf_boosts:
                score = (score * 0.7) + (sf_boosts[rk] * 0.3)

            z_score = (score - mean) / std
            margin = score - med
            scored.append({
                "id": rule.get("id", rk),
                "rule_key": rk,
                "source": str(rule.get("source") or ""),
                "title": title,
                "keywords": _parse_keywords(rule.get("keywords")),
                "response": str(rule.get("response") or ""),
                "is_handoff": bool(rule.get("is_handoff")),
                "priority": int(rule.get("priority") or 0),
                "raw_cosine": round(raw_cos, 4),
                "z_score": round(z_score, 3),
                "margin": round(margin, 4),
                "lex_hits": lex_hits,
                "score": round(score, 4),
            })

        scored.sort(key=lambda item: (item["score"], item["lex_hits"], item["priority"]), reverse=True)
        seen_titles = set()
        top_items: List[Dict[str, Any]] = []
        for item in scored:
            tn = item["title"].strip().lower()
            if tn in seen_titles:
                continue
            seen_titles.add(tn)
            top_items.append(item)
            if len(top_items) >= max(1, top_k):
                break

        best = top_items[0] if top_items else None
        decision = "miss"
        if best:
            req_direct = handoff_threshold if best["is_handoff"] else direct_threshold
            if len(self.rules) >= 20:
                direct_ok = (
                    best["score"] >= req_direct
                    and best["margin"] >= min_margin
                    and (best["z_score"] >= min_z_score or best["lex_hits"] >= 1)
                )
                suggest_ok = (
                    best["score"] >= suggest_threshold
                    and best["margin"] >= (min_margin * 0.65)
                    and (best["z_score"] >= min_suggest_z_score or best["lex_hits"] >= 1)
                )
                if direct_ok:
                    decision = "direct"
                elif suggest_ok:
                    decision = "suggest"
            else:
                if best["score"] >= req_direct:
                    decision = "direct"
                elif best["score"] >= suggest_threshold:
                    decision = "suggest"

        latency = round((time.perf_counter() - started) * 1000, 1)
        return {
            "ok": True,
            "mode": mode,
            "decision": decision,
            "best_match": best if decision != "miss" else None,
            "suggestions": [s for s in top_items if s["score"] >= suggest_threshold] if decision != "direct" else [],
            "top_candidates": [
                {
                    "rule_key": x["rule_key"],
                    "title": x["title"],
                    "score": x["score"],
                    "raw_cosine": x["raw_cosine"],
                    "z_score": x["z_score"],
                    "margin": x["margin"],
                    "lex_hits": x["lex_hits"],
                }
                for x in top_items
            ],
            "latency_ms": latency,
        }


def create_app():
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field

    api_key = os.getenv("ENGINE_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL).strip()
    data_dir = os.getenv("DATA_DIR", "/data").strip()
    semantic_mode = os.getenv("SEMANTIC_MODE", "shadow").strip().lower()
    if semantic_mode not in {"off", "shadow", "live"}:
        semantic_mode = "shadow"
    direct_th = float(os.getenv("DIRECT_THRESHOLD", "0.84"))
    suggest_th = float(os.getenv("SUGGEST_THRESHOLD", "0.76"))
    handoff_th = float(os.getenv("HANDOFF_THRESHOLD", "0.88"))
    min_z = float(os.getenv("MIN_Z_SCORE", "2.0"))
    min_suggest_z = float(os.getenv("MIN_SUGGEST_Z_SCORE", "1.4"))
    min_margin = float(os.getenv("MIN_MARGIN", "0.045"))
    setfit_dir = os.getenv("SETFIT_MODEL_DIR", "").strip()

    def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if not api_key:
            return
        expected = f"Bearer {api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    state: Dict[str, Any] = {"index": None, "model_name": model_name, "mode": semantic_mode}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        embedder = load_embedder(model_name)
        idx = SemanticIndex(
            data_dir=data_dir,
            embedder=embedder,
            use_e5_prefix=("e5" in model_name.lower()),
        )
        if setfit_dir and Path(setfit_dir).exists():
            try:
                from setfit import SetFitModel

                idx.setfit_model = SetFitModel.from_pretrained(setfit_dir)
            except Exception:
                idx.setfit_model = None
        state["index"] = idx
        yield

    app = FastAPI(title="BDWebs Semantic Bot Engine", version="1.1.0", lifespan=lifespan)

    class MatchRequest(BaseModel):
        query: str = Field(..., min_length=1)
        active_topic: str = ""
        channel: str = "Channel::WebWidget"
        top_k: int = 3
        direct_threshold: Optional[float] = None
        suggest_threshold: Optional[float] = None

    class ReindexRequest(BaseModel):
        rules: List[dict]

    @app.get("/health")
    def health() -> Dict[str, Any]:
        idx: Optional[SemanticIndex] = state.get("index")
        return {
            "ok": True,
            "semantic_mode": state["mode"],
            "model_name": state["model_name"],
            "rule_count": len(idx.rules) if idx else 0,
            "index_ready": bool(idx and idx.vectors is not None),
            "setfit_loaded": bool(idx and idx.setfit_model is not None),
        }

    @app.post("/reindex", dependencies=[Depends(require_auth)])
    def reindex(req: ReindexRequest) -> Dict[str, Any]:
        idx: SemanticIndex = state["index"]
        return idx.reindex(req.rules)

    @app.post("/match", dependencies=[Depends(require_auth)])
    def match(req: MatchRequest) -> Dict[str, Any]:
        idx: SemanticIndex = state["index"]
        return idx.match(
            req.query,
            mode=state["mode"],
            active_topic=req.active_topic,
            top_k=req.top_k,
            direct_threshold=req.direct_threshold if req.direct_threshold is not None else direct_th,
            suggest_threshold=req.suggest_threshold if req.suggest_threshold is not None else suggest_th,
            handoff_threshold=handoff_th,
            min_z_score=min_z,
            min_suggest_z_score=min_suggest_z,
            min_margin=min_margin,
        )

    return app


try:
    app = create_app()
except ImportError:
    app = None
