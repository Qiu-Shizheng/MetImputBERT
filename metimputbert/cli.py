#!/usr/bin/env python
import argparse
import os
import sys
import logging
from metimputbert import Imputer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="MetImputBERT: Transformer-based imputation tool for metabolomics data.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file with missing data.")
    parser.add_argument("--model", required=True, choices=["168", "249"], help="Model type: '168' or '249'.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} does not exist.")
        sys.exit(1)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Selected model type: {args.model}")

    try:
        imputer = Imputer(model_type=args.model)
        imputed_df = imputer.impute(args.input)
        base, ext = os.path.splitext(args.input)
        output_csv = base + "_imputed" + ext
        imputed_df.to_csv(output_csv, index=False)
        logger.info(f"Imputation completed and saved as {output_csv}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()