from pydantic import BaseModel, Field, ConfigDict, AliasChoices

class ComplexityCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    length_thresholds: list[int] = Field(min_items=2)
    branching_markers: list[str]
    uniq_ratio_penalty_lt: float
    multistep_scale: float

class LatencyKeywordsCfg(BaseModel):
    medium: list[str]

class FeatureCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str

    json:      list[str] = Field(validation_alias=AliasChoices("json"))
    code:      list[str] = Field(validation_alias=AliasChoices("code"))
    multistep: list[str] = Field(validation_alias=AliasChoices("multistep"))
    urgency:   list[str] = Field(validation_alias=AliasChoices("urgency"))
    safety:    list[str] = Field(validation_alias=AliasChoices("safety"))

    latency: LatencyCfg = Field(validation_alias=AliasChoices("latency", "latency_keywords"))
    tiktoken_encoding: str | None = None
    complexity: ComplexityCfg