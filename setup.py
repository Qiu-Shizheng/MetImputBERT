from setuptools import setup, find_packages

setup(
    name="MetImputBERT",
    version="0.1.0",
    description="Transformer-based imputation tool for metabolomics data",
    author="Shizheng Qiu",
    author_email="qiushizheng@hit.edu.cn",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "transformers"
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "metimputbert=metimputbert.cli:main"
        ]
    },
)
