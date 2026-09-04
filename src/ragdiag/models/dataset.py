"""Domain model representing a golden evaluation dataset."""

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from ragdiag.models.sample import QuerySample


class GoldenDataset(BaseModel):
    """A validated golden evaluation dataset containing query samples.

    Attributes:
        name: Name of the dataset (e.g., 'payment_gateway_eval_v1').
        version: Version string (e.g., '1.0').
        samples: List of at least one validated `QuerySample`.
    """

    name: str
    version: str
    samples: list[QuerySample] = Field(min_length=1)

    @field_validator("name", "version", mode="after")
    @classmethod
    def validate_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        stripped = v.strip()
        if not stripped:
            field_name = info.field_name or "field"
            raise ValueError(f"Field '{field_name}' must not be empty or whitespace-only.")
        return stripped

    @model_validator(mode="after")
    def validate_unique_sample_ids(self) -> "GoldenDataset":
        seen: set[str] = set()
        duplicates: list[str] = []
        for sample in self.samples:
            if sample.id in seen:
                duplicates.append(sample.id)
            seen.add(sample.id)
        if duplicates:
            unique_dups = sorted(set(duplicates))
            raise ValueError(f"Dataset contains duplicate sample IDs: {', '.join(unique_dups)}")
        return self
