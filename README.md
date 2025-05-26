# MetImputBERT

MetImputBERT is a Linux tool for imputing missing values in NMR metabolomics data using a pretrained transformer-based model.

## Features

- Two pretrained models:
  - 168 model: Expects a CSV file with 169 columns (first column is "eid", remaining 168 columns are metabolite features).
  - 249 model: Expects a CSV file with 250 columns (first column is "eid", next 168 columns are metabolite features, and remaining 81 columns are ratios).
- Automatically computes per-column normalization parameters from the input data.
- Performs imputation and then rescales the output back to the original scale.

## System requirements
torch 2.4.1+cu124  
python 3.11.9  
pytorch-cuda 12.4    
numpy 1.26.4

## Installation
Clone the repository and install:

```bash
git clone https://github.com/Qiu-Shizheng/MetImputBERT.git
cd MetImputBERT
pip install .
```
## Usage
Run the tool with the command:

```bash
metimputbert --input example.csv --model 249
```
The imputed output is saved as example_imputed.csv.
