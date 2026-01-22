from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class AssetPaths:
    weights_path: Path
    scaler_path: Path
    columns_path: Path


def get_default_assets_249() -> AssetPaths:

    pkg = "metimputbert"

    weights = resources.files(pkg).joinpath("weights").joinpath("249_model.pt")
    scaler = resources.files(pkg).joinpath("assets").joinpath("249").joinpath("scaler.npz")
    columns = resources.files(pkg).joinpath("assets").joinpath("249").joinpath("columns.json")

    return AssetPaths(weights_path=weights, scaler_path=scaler, columns_path=columns)
