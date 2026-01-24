# MetImputBERT




MetImputBERT is a python tool for imputing missing values in NMR metabolomics data using a pretrained transformer-based model.

## Features

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
pip install -e .    
```
If you are unable to obtain the model weights from GitHub, you can also download them via figshare:https://doi.org/10.6084/m9.figshare.30744218
        
        
        
        
        
   
        

## Features
- Input: raw (not standardized) metabolite table with missing values.
- Output: full table with missing values imputed.
- Non-missing values remain unchanged.

        
        

```bash
metimputbert \
  -i input.csv \
  -o output_imputed.csv \
  --eid_col eid \
  --batch_size 64 \
  --clip_nonneg
```
The imputed output is saved as output_imputed.csv.
```bash
Saved imputed file to: /home/user/output_imputed.csv
Missing count imputed: 50
Missing ratio: 0.00001
```


