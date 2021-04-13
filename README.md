# PredOps

PredOps is a forecasting tool with the following features:

- [x] Support for multiple forecast methods:
    - [x] Baselines
    - [ ] Time series statistical methods
    - [x] Forest ensemble algorithms
    - [ ] Deep Learning approaches

- [x] Support for multiple remote compute environments
    - [x] local environment
    - [ ] Azure ML 
    - [x] SAS Viya server

- [x] Back-testing
- [x] Segmentation capabilities
- [x] Time segmentation for a more precise forecast

The project will implement the following MLOps practices:
- [x] Experiment tracking
- [ ] Performance monitoring
- [ ] Retraining
- [ ] CI/CD/CT

## Installation

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
- [Kaggle discussion](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163216)
- Best practices of developement from [madewithml repository](https://github.com/GokuMohandas/applied-ml)