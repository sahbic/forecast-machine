# forecast-machine

This aims to be a forecast tool, with the following features:
- Various forecast methods:
    - Time series statistical methods
    - Forest ensemble algorithms
    - Deep Learning approaches
- Back-testing
- Segmentation capabilities
- Time segmentation for a more precise forecast

## Install dependencies

Install from requirements
```
pip install -r requirements.txt
```

or from scratch

```
conda create -n machine python=3.7
pip install swat python-dotenv scikit-learn python-slugify openpyxl
```

## How to run
```
conda activate machine
```

```
python ./src/main.py 
````

or with parameters:
```
python src/main.py -pr "tomahawk" -dv "indicateur" -sva "canal" -npg "3" -flc "2017-06,2017-12"
```

```
python src/main.py -pr "m5a" -dv "Quantity" -flc "31-10-2015" -npr "28" -npg "7" -sva "store_id"
```

## References

- [M5 Competition 4th place](https://github.com/monsaraida/kaggle-m5-forecasting-accuracy-4th-place)