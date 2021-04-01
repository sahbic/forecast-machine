# PredOps

PredOps is a forecasting tool with the following features:

- Various forecast methods:
    - Time series statistical methods
    - Forest ensemble algorithms
    - Deep Learning approaches
- Ability to call multiple remote compute environments
- Back-testing
- Segmentation capabilities
- Time segmentation for a more precise forecast

The project will implement the following MLOps practices:
- Experiment tracking
- Performance monitoring
- Retraining
- CI/CD/CT

## Installation

<!-- Install from requirements
```bash
pip install -r requirements.txt
```

or from scratch

```bash
conda create -n machine python=3.7
pip install swat python-dotenv scikit-learn python-slugify openpyxl
pip install typer
pip install kaggle
pip install pretty-errors
``` -->

```bash
python3 -m venv venv
source venv/bin/activate
# On windows powershell
# .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

Installation of the development environment:
```bash
python -m pip install -e ".[dev]"
```

## How to run

<!-- ```bash
conda activate machine
```

```bash
python ./src/main.py 
```

or with parameters:
```bash
python src/main.py -pr "tstest" -dv "indicateur" -sva "canal" -npg "3" -flc "2017-06,2017-12"
```

```bash
# train models (hypertuning and cross-evaluation)
python src/main.py -pr "m5a" -dv "Quantity" -nfo "3" -npr "28" -npg "7" -sva "store_id" -mod "tune_train_eval"
# backtest previously trained models
python src/main.py -pr "m5a" -dv "Quantity" -nfo "3" -npr "28" -npg "7" -sva "store_id" -mod "backtest"
``` -->

### CLI

#### Example: M5 Project

1. Download data
```bash
predops download-data m5a
```
2. Generate base file (with sample option)
```bash
predops generate-base-file m5a --sample
```
3. Train models: search, train and evaluate
```bash
# default parameters
predops train m5a --target Quantity
# advanced parameters
predops train m5a --target Quantity --number-predictions 28 --column-segment-groupby store_id --n-predictions-groupby 7
```

### MLFlow

MLFlow allows to track the experiments

```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

## References

- [M5 Competition 4th place](https://github.com/monsaraida/kaggle-m5-forecasting-accuracy-4th-place)
- Best practices of developement fron [madewithml repository](https://github.com/GokuMohandas/applied-ml)