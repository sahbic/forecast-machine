# forecast-machine

This aims to be a forecast tool, with the following features:

- Various forecast methods:
    - Time series statistical methods
    - Forest ensemble algorithms
    - Deep Learning approaches
- Ability to call multiple compute environments
- Back-testing
- Segmentation capabilities
- Time segmentation for a more precise forecast

## Install dependencies

Install from requirements
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
```

## How to run
```bash
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
```

## References

- [M5 Competition 4th place](https://github.com/monsaraida/kaggle-m5-forecasting-accuracy-4th-place)
- Best practices of developement fron [madewithml repository](https://github.com/GokuMohandas/applied-ml)