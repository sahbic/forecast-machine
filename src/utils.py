import datetime
import logging
from pathlib import Path


def get_logger(log_name, log_dir_path, log_level=logging.DEBUG):
    logger = logging.getLogger(log_name)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file_path = str(
        log_dir_path / (log_name + datetime.datetime.today().strftime("_%Y_%m%d.log"))
    )
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(log_level)
    return logger


class Log(object):
    def __init__(self, logger):
        self.logger = logger

    def info(self, *messages):
        return self.logger.info(Log.format_message(messages))

    def debug(self, *messages):
        return self.logger.debug(Log.format_message(messages))

    def warning(self, *messages):
        return self.logger.warning(Log.format_message(messages))

    def error(self, *messages):
        return self.logger.error(Log.format_message(messages))

    def exception(self, *messages):
        return self.logger.exception(Log.format_message(messages))

    @staticmethod
    def format_message(messages):
        if len(messages) == 1 and isinstance(messages[0], list):
            messages = tuple(messages[0])
        return "\t".join(map(str, messages))


class Params(object):
    def __init__(self, setting):
        self.setting = setting
        self.data_dir_path = Path(setting["data_dir_path"])

        self.project_key = setting["project_key"]
        self.n_folds = int(setting["n_folds"])

        self.raw_dir_path = self.data_dir_path / "raw" / setting["project_key"]
        self.raw_dir_path.mkdir(parents=True, exist_ok=True)

        self.output_name = Path(setting["output_name"])
        self.output_dir_path = (
            self.data_dir_path / "output" / self.output_name / setting["project_key"]
        )
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        self.log_name = setting["log_name"]
        self.log_dir_path = self.output_dir_path / "log"
        self.log_dir_path.mkdir(parents=True, exist_ok=True)
        self.log = Log(get_logger(self.log_name, self.log_dir_path))

        self.mode = setting["mode"]
        if self.mode not in ["tune_train_eval", "train_eval", "predict", "backtest"]:
            self.log.error(
                "Mode value is not in ['tune_train_eval','train_eval','predict','backtest']"
            )
            raise ValueError

        self.result_dir_path = self.output_dir_path / "result"
        self.result_dir_path.mkdir(parents=True, exist_ok=True)

        self.work_dir_path = self.output_dir_path / "work"
        self.work_dir_path.mkdir(parents=True, exist_ok=True)

        self.time_col = setting["time_col"]
        self.id_col = setting["id_col"]
        self.dependent_var = setting["dependent_var"]
        self.number_predictions = int(setting["number_predictions"])
        self.n_predictions_groupby = int(setting["n_predictions_groupby"])

        self.segment_groupby_column = setting["segment_groupby_column"]

        if self.number_predictions % self.n_predictions_groupby != 0:
            self.log.warning(
                "number_predictions must be a multiple of n_predictions_groupby. n_predictions_groupby is set to be equal to number_predictions (no time grouping)"
            )
            self.n_predictions_groupby = self.number_predictions

        ratio_prediction = int(self.number_predictions / self.n_predictions_groupby)
        self.prediction_horizon_list = (
            [i * self.n_predictions_groupby for i in range(1, ratio_prediction + 1)]
            if (ratio_prediction > 1)
            else [self.number_predictions]
        )

        self.model_dir_path = self.output_dir_path / "model"
        self.model_dir_path.mkdir(parents=True, exist_ok=True)

        self.end_train_x_list = [str(fold_id) for fold_id in setting["fold_id_list_csv"].split(",")]

        return
