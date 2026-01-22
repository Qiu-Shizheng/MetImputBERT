from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from importlib import resources

from .assets import get_default_assets_249
from .model import MetaboliteBERTModel


@dataclass
class ImputerConfig:
    num_metabolites: int = 249
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 8
    dropout: float = 0.1


    batch_size: int = 1024
    clip_nonneg: bool = False  
    device: Optional[str] = None  # None => auto


class ZScoreScaler:

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[self.std == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean

    @classmethod
    def load(cls, path: str):
        d = np.load(path)
        return cls(d["mean"], d["std"])


class MetImputBERTImputer:
    """
    推理端：对 249 代谢物表格进行缺失插补（只填缺失位）。
    输入为原始空间（未标准化），内部用训练 scaler 标准化，模型预测后反标准化。
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        columns_path: Optional[str] = None,
        config: Optional[ImputerConfig] = None,
    ):
        self.cfg = config or ImputerConfig()

        # 资源路径：若用户未指定，则使用包内默认
        self._default_assets = get_default_assets_249()
        self.weights_path = weights_path
        self.scaler_path = scaler_path
        self.columns_path = columns_path

        self.device = self._resolve_device(self.cfg.device)
        self.model = None
        self.scaler = None
        self.columns = None

        self._load_all()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_all(self):

        weights_tr = self._default_assets.weights_path if self.weights_path is None else self.weights_path
        scaler_tr = self._default_assets.scaler_path if self.scaler_path is None else self.scaler_path
        columns_tr = self._default_assets.columns_path if self.columns_path is None else self.columns_path


        def _as_path(tr):
            if isinstance(tr, str):
                return None, tr
            # Traversable
            ctx = resources.as_file(tr)
            p = str(ctx.__enter__())
            return ctx, p

        ctxs = []
        try:
            c1, weights_path = _as_path(weights_tr);  ctxs.append(c1)
            c2, scaler_path = _as_path(scaler_tr);    ctxs.append(c2)
            c3, columns_path = _as_path(columns_tr);  ctxs.append(c3)

            # load columns
            with open(columns_path, "r", encoding="utf-8") as f:
                columns = json.load(f)
            if not isinstance(columns, list) or len(columns) != self.cfg.num_metabolites:
                raise ValueError(f"columns.json must be a list of length {self.cfg.num_metabolites}")

            # load scaler
            scaler = ZScoreScaler.load(scaler_path)
            if scaler.mean.shape[0] != self.cfg.num_metabolites:
                raise ValueError("scaler mean dimension mismatch with num_metabolites")

            # build model
            model = MetaboliteBERTModel(
                num_metabolites=self.cfg.num_metabolites,
                hidden_size=self.cfg.hidden_size,
                num_layers=self.cfg.num_layers,
                num_attention_heads=self.cfg.num_attention_heads,
                dropout=self.cfg.dropout,
            )
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=True)
            model.to(self.device)
            model.eval()

            self.columns = columns
            self.scaler = scaler
            self.model = model
        finally:

            for c in ctxs:
                if c is not None:
                    c.__exit__(None, None, None)

    def _align_features(self, df: pd.DataFrame, eid_col: Optional[str], strict: bool = True) -> pd.DataFrame:

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input file missing required metabolite columns (count={len(missing)}). Example: {missing[:5]}")

        feat_df = df[self.columns].copy()


        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Internal columns list has duplicates (unexpected).")

        return feat_df

    @torch.no_grad()
    def impute_dataframe(
        self,
        df: pd.DataFrame,
        eid_col: Optional[str] = "eid",
        strict_columns: bool = True,
        return_mask_stats: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict]]:

        if self.model is None or self.scaler is None or self.columns is None:
            raise RuntimeError("Imputer not initialized correctly.")

        
        has_eid = eid_col is not None and eid_col in df.columns

        feat_df = self._align_features(df, eid_col=eid_col, strict=strict_columns)

        feat_numeric = feat_df.apply(pd.to_numeric, errors="coerce")  # 推理端用 coerce 更友好
        X = feat_numeric.values.astype(np.float32)  # shape [N,249]

        missing = np.isnan(X)
        observed = ~missing
        n_missing = int(missing.sum())
        n_total = int(X.size)


        if n_missing == 0:
            if return_mask_stats:
                return df.copy(), {"missing_count": 0, "missing_ratio": 0.0}
            return df.copy()


        X_filled = np.where(missing, self.scaler.mean[None, :], X).astype(np.float32)

        Z = self.scaler.transform(X_filled)

        # observed_mask: 1=observed, 0=missing (torch long)
        observed_mask = observed.astype(np.int64)


        N = Z.shape[0]
        bs = int(self.cfg.batch_size)
        Z_pred = np.empty_like(Z, dtype=np.float32)

        for start in range(0, N, bs):
            end = min(N, start + bs)
            z_batch = torch.from_numpy(Z[start:end]).to(self.device)
            m_batch = torch.from_numpy(observed_mask[start:end]).to(self.device)

            z_hat, _ = self.model(z_batch, m_batch, output_attentions=False)  # [b,249]
            Z_pred[start:end] = z_hat.detach().to("cpu").numpy().astype(np.float32)


        X_pred = self.scaler.inverse_transform(Z_pred)


        X_out = X.copy()
        X_out[missing] = X_pred[missing]


        if self.cfg.clip_nonneg:
            X_out[missing] = np.clip(X_out[missing], 0.0, np.inf)

        
        out_df = df.copy()
        out_feat = pd.DataFrame(X_out, columns=self.columns, index=out_df.index)
        for c in self.columns:
            out_df[c] = out_feat[c]

        if return_mask_stats:
            return out_df, {"missing_count": n_missing, "missing_ratio": n_missing / n_total}

        return out_df
