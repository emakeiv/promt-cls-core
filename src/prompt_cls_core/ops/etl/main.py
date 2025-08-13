from __future__ import annotations

import yaml  
import tiktoken 
import pandas as pd
import re, os, json, hashlib

from pydantic import ValidationError
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Any

from src.core.etl.schema import FeatureCfg

@dataclass
class Features:
    char_len: int
    word_len: int
    token_est: int
    line_count: int
    has_json: bool
    strict_json: bool
    has_code: bool
    multistep_score: int
    complexity_score: float
    latency_tier: str
    safety_sensitive: bool

class FeatureExtractor:
    '''
    '''

    def __init__(self, cfg: FeatureCfg):
        if cfg is None:
            raise ValueError("FeatureCfg is required. Load via FeatureExtractor.from_file(...)")
        
        self.token = None

        self.multistep = tuple(s.lower() for s in cfg.multistep)
        self.latency_medium = tuple(s.lower() for s in cfg.latency.medium)
        self.urgency = tuple(s.lower() for s in cfg.urgency)
        self.safety = tuple(s.lower() for s in cfg.safety)
        self.json_terms = tuple(s.lower() for s in cfg.json_terms)
        self.code_terms = tuple(s.lower() for s in cfg.code_terms)
        self.complexity = cfg.complexity
        self.cx = cfg.complexity.model_dump()

        self.tk_enc = None
        enc = (cfg.tiktoken_encoding or "").strip()
        if enc and tiktoken:
            try:
                self.tk_enc = tiktoken.get_encoding(enc)
            except Exception:
                self.tk_enc = None
        
        self.config_hash = self._hash_config(cfg.model_dump())
        self.extractor_version = "fe-1.0"

    @staticmethod
    def _hash_config(cfg_dict: Dict) -> str:
        blob = json.dumps(cfg_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    @staticmethod
    def _id_for_prompt(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"feature extractor config file not found: {path}")
        if path.endswith((".yml", ".yaml")):
            if yaml is None:
                raise RuntimeError("pyyaml not installed (uv add pyyaml).")
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        try:
            cfg = FeatureCfg.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"Invalid feature config: {e}") from e
        return cls(cfg)

    def _get_tokens(self, text: str) -> list[str]:
        return (text).casefold().translate(self._punct_table).split()
        
    def _estimate_tokens(self, text: str) -> int:
        if self.tk_enc:
            try:
                return len(self.tk_enc.encode(text))
            except Exception:
                pass
        return len(self.token.findall(text))

    def _score_complexity(self, text: str) -> float:
        tokens = self._get_tokens(text)
        n = len(tokens)
        if n == 0:
            return 0.0
        
        score = float(n > self.complexity.length_thresholds)
        tokset = set(tokens)
        score += sum(1.0 for m in self._branch_set if m in tokset)
        score += sum(1.0 for m in self._multistep_set if m in tokset) * self._multistep_scale

        uniq_ratio = len(tokset) / n
        if uniq_ratio < self._uniq_penalty_lt:
            score += 1.0

        return score
    
    def _estimate_latency(self, text: str) -> str:
        pass

    def _get_one(self, text: str) -> Features:
        pass

    def _get_batch(self, prompts: Iterable[str], ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        pass

    def save(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            df.to_parquet(path, index=False)
        elif ext in (".feather", ".ft"):
            df.to_feather(path)
        else:
            raise ValueError("use .parquet or .feather")

    def load(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        elif ext in (".feather", ".ft"):
            return pd.read_feather(path)
        else:
            raise ValueError("use .parquet or .feather")