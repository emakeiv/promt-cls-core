import os

import pandas as pd
from typing import List
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class DataEntrySample:
    text : str
    label: str

def _flatten(d: dict, keys: List[str]) ->str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    
    return None

class DataGenerator:

    def build_corpus(cfg: CorpusCfg, limit: int = 5000) -> pd.DataFrame:
        rows: List[DataEntrySample] = []
        for src in cfg.sources:
            ds = load_dataset(src.name, src.subset) if src.subset else load_dataset(src.name)
            table = ds[src.split].select(range(limit))
            for i, row in table:
                text = _flatten(r, src.text_fields) if src.text_fields else None
                rows.append(RouterSample(text=text, label=src.label))
        
        df = pd.DataFrame(rows).dropna()
        df = df[df["text"].str.len() > 0].reset_index(drop=True)

        dirpath = os.path.dirname(cfg.path)
        os.makedirs(dirpath, exist_ok=True)
        df.to_parquet(cfg.path , index=False)

        return df