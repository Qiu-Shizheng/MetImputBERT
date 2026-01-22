from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def infer_sep(path: str) -> str:

    p = Path(path)
    suf = p.suffix.lower()
    if suf in [".tsv", ".txt"]:
        return "\t"
    return ","


def read_table(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    sep = sep if sep is not None else infer_sep(path)
    df = pd.read_csv(path, sep=sep)
    return df


def write_table(df: pd.DataFrame, path: str, sep: Optional[str] = None):
    sep = sep if sep is not None else infer_sep(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False)


def to_float_frame(df: pd.DataFrame) -> pd.DataFrame:

    return df.apply(pd.to_numeric, errors="raise")


def nan_mask(X: np.ndarray) -> np.ndarray:
    
    return np.isnan(X)
