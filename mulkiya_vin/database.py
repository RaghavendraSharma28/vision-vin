from __future__ import annotations

from pathlib import Path
from typing import Any

from .vin import normalize_vin_text


class VehicleDatabase:
    def lookup(self, vin: str) -> dict[str, Any] | None:
        raise NotImplementedError


class DataFrameVehicleDatabase(VehicleDatabase):
    def __init__(self, frame, vin_column: str = "VIN Number") -> None:
        self.vin_column = vin_column
        self.frame = frame.copy()
        if vin_column not in self.frame.columns:
            raise ValueError(f"Missing VIN column: {vin_column}")
        self.frame["_normalized_vin"] = self.frame[vin_column].map(lambda value: normalize_vin_text(str(value)))

    @classmethod
    def from_file(cls, path: str | Path, vin_column: str = "VIN Number") -> "DataFrameVehicleDatabase":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install database extras with: pip install -e .[db]") from exc

        path = Path(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported database file: {path}")
        return cls(frame, vin_column=vin_column)

    def lookup(self, vin: str) -> dict[str, Any] | None:
        normalized = normalize_vin_text(vin)
        matches = self.frame[self.frame["_normalized_vin"] == normalized]
        if matches.empty:
            return None
        record = matches.iloc[0].drop(labels=["_normalized_vin"]).to_dict()
        return {str(key): _json_safe(value) for key, value in record.items()}


class EmptyVehicleDatabase(VehicleDatabase):
    def lookup(self, vin: str) -> dict[str, Any] | None:
        return None


def _json_safe(value: Any) -> Any:
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and pd.isna(value):
        return None
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
