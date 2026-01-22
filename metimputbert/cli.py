from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .imputer import MetImputBERTImputer, ImputerConfig
from .io_utils import read_table, write_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="metimputbert",
        description="MetImputBERT: impute missing metabolite values (only missing positions are filled).",
    )
    p.add_argument("-i", "--input", required=True, help="Input table path (.csv/.tsv). Contains eid + 249 metabolites.")
    p.add_argument("-o", "--output", required=True, help="Output path (.csv/.tsv).")

    p.add_argument("--sep", default=None, help="Delimiter. Default inferred from file extension.")
    p.add_argument("--eid_col", default="eid", help="EID column name to keep (not imputed). Default: eid")

    p.add_argument("--device", default=None, help="Device string, e.g. cuda:0 or cpu. Default: auto.")
    p.add_argument("--batch_size", type=int, default=1024, help="Inference batch size. Default: 1024")
    p.add_argument("--clip_nonneg", action="store_true", help="Clip imputed values (missing positions only) to >=0.")

    # Advanced: allow external assets directory
    p.add_argument(
        "--assets_dir",
        default=None,
        help=(
            "Optional directory containing columns.json and scaler.npz (and optionally weights). "
            "If set, will look for: assets_dir/columns.json and assets_dir/scaler.npz."
        ),
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Optional path to weights .pt. If not set, use package default weights/249_model.pt.",
    )
    p.add_argument("--scaler", default=None, help="Optional path to scaler.npz. Overrides assets_dir/package default.")
    p.add_argument("--columns", default=None, help="Optional path to columns.json. Overrides assets_dir/package default.")

    return p


def main():
    args = build_parser().parse_args()

    inp = args.input
    out = args.output

    # resolve scaler/columns from assets_dir if provided
    scaler = args.scaler
    columns = args.columns
    if args.assets_dir is not None:
        ad = Path(args.assets_dir)
        if scaler is None:
            scaler = str(ad / "scaler.npz")
        if columns is None:
            columns = str(ad / "columns.json")

    cfg = ImputerConfig(
        batch_size=args.batch_size,
        clip_nonneg=args.clip_nonneg,
        device=args.device,
    )

    imputer = MetImputBERTImputer(
        weights_path=args.weights,
        scaler_path=scaler,
        columns_path=columns,
        config=cfg,
    )

    df = read_table(inp, sep=args.sep)

    out_df, stats = imputer.impute_dataframe(
        df,
        eid_col=args.eid_col if args.eid_col else None,
        strict_columns=True,
        return_mask_stats=True,
    )


    write_table(out_df, out, sep=args.sep)


    print(f"Saved imputed file to: {out}")
    print(f"Missing count imputed: {stats['missing_count']}")
    print(f"Missing ratio: {stats['missing_ratio']:.6f}")


if __name__ == "__main__":
    main()
