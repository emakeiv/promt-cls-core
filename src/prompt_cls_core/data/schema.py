
from pydantic import BaseModel, Field
from typing import List, Optional

class SourceCfg(BaseModel):
    name: str  
    subset: Optional[str] = None
    split: str = "train"
    text_fields: List[str] = Field(default_factory=list)
    label: Label
    template: Optional[str] = None

class CorpusCfg(BaseModel):
    sources: List[SourceCfg]
    balance: bool = True 
    path: Optional[str] = "./dataset/router_corpus.parquet"