# MetImputBERT

MetImputBERT is a python tool for imputing missing values in NMR metabolomics data using a pretrained transformer-based model.

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
The weight of pre-training model is also provided at Figshare.

## Usage
For the 168 model, the default batch size is set to 36, which is suitable for the RTX2080Ti with 11GB. If you have a GPU with higher memory, such as the A100 (40GB), you can set the batch size in imputer.py to 192. Run the tool with the command:

```bash
metimputbert --input example.csv --model 249
metimputbert --input example2.csv --model 168
```
The imputed output is saved as example_imputed.csv.
```bash
[INFO] Input file: /home/user/MetImputBERT/example2.csv
[INFO] Selected model type: 168
[INFO] Using local weight file: /home/user/MetImputBERT/metimputbert/weights/168_model.pt
[INFO] Loaded pretrained model from /home/user/metimputbert/weights/168_model.pt on cuda
[INFO] Loading data from /home/user/MetImputBERT/example2.csv
[INFO] Normalizing data
[INFO] Performing imputation
[INFO] De-normalizing data
[INFO] Imputation complete
[INFO] Imputation completed and saved as /home/user/MetImputBERT/example2_imputed.csv
```
