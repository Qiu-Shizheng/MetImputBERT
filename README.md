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
The weight of pre-training model is also provided at Figshare: https://figshare.com/s/bfb68f0387a5b430eacd

## Usage
For the 168 model, the default batch size is set to 36.


| Batch Size | 	Missing Count	| Runtime (s)	| Max Memory (MB) |
| --- | --- | --- | --- |
|32	| 10000	| 3.86 | 851 |
|64	| 10000	| 3.53	| 1,043 |
|96	| 10000	| 3.44 | 1,235 |
|256	| 10000	| 3.22 	| 2,191 |
|512	| 10000	| 3.10 	| 3,722 |



| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |


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
