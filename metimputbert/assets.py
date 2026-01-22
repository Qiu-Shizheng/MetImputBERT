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
    """
    返回包内默认 249 模型的资源路径：
    - weights/249_model.pt
    - assets/249/scaler.npz
    - assets/249/columns.json

    注意：这些文件需要被打包进 wheel/sdist（见 pyproject.toml）。
    """
    pkg = "metimputbert"

    weights = resources.files(pkg).joinpath("weights").joinpath("249_model.pt")
    scaler = resources.files(pkg).joinpath("assets").joinpath("249").joinpath("scaler.npz")
    columns = resources.files(pkg).joinpath("assets").joinpath("249").joinpath("columns.json")

    # resources 返回 Traversable，这里转成本地 Path（对 wheel 也可用 as_file）
    # 为了兼容 zipimport，这里不直接强转 Path，而在调用方用 as_file 临时落地。
    return AssetPaths(weights_path=weights, scaler_path=scaler, columns_path=columns)