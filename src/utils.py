import logging
import datetime
from pathlib import Path

def get_logger(log_name, log_dir_path, log_level=logging.DEBUG):
    logger = logging.getLogger(log_name)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file_path = str(log_dir_path / (log_name + datetime.datetime.today().strftime('_%Y_%m%d.log')))
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
        return '\t'.join(map(str, messages))

    # def log_evaluation(self, period=100, show_stdv=True, level=logging.INFO):
    #     def _callback(env):
    #         if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
    #             result = '\t'.join(
    #                 [lgb.callback._format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
    #             self.logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))

    #     _callback.order = 10
    #     return _callback


class Params(object):
    def __init__(self, setting):
        self.setting = setting
        self.data_dir_path = Path(setting['data_dir_path'])

        self.raw_dir_path = self.data_dir_path / 'raw' / setting['project_key']
        self.raw_dir_path.mkdir(parents=True, exist_ok=True)

        self.output_name = Path(setting['output_name'])
        self.output_dir_path = self.data_dir_path / 'output' / self.output_name
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        self.log_name = setting['log_name']
        self.log_dir_path = self.output_dir_path / 'log'
        self.log_dir_path.mkdir(parents=True, exist_ok=True)
        self.log = Log(get_logger(self.log_name, self.log_dir_path))

        self.result_dir_path = self.output_dir_path / 'result'
        self.result_dir_path.mkdir(parents=True, exist_ok=True)

        self.work_dir_path = self.output_dir_path / 'work'
        self.work_dir_path.mkdir(parents=True, exist_ok=True)

        self.time_col = setting['time_col']
        self.id_col = setting['id_col']
        self.dependent_var = setting['dependent_var']
        self.number_predictions = int(setting['number_predictions'])
        self.n_predictions_groupby = int(setting['n_predictions_groupby'])
        self.segment_groupby_column = setting['segment_groupby_column']

        if self.number_predictions % self.n_predictions_groupby != 0:
            self.log.warning("number_predictions must be a multiple of n_predictions_groupby. n_predictions_groupby is set to be equal to number_predictions (no time grouping)")
            self.n_predictions_groupby = self.number_predictions
        
        ratio_prediction = int(self.number_predictions/self.n_predictions_groupby)
        self.prediction_horizon_list = [i*self.n_predictions_groupby for i in range(1,ratio_prediction + 1)] if (ratio_prediction > 1) else [self.number_predictions]

        self.model_dir_path = self.output_dir_path / 'model'
        self.model_dir_path.mkdir(parents=True, exist_ok=True)

        # self.seed = 42
        # Util.set_seed(self.seed)

        # self.sampling_rate = setting['sampling_rate']
        # self.export_all_flag = False
        # self.recursive_feature_flag = False

        # self.target = 'sales'
        # self.start_train_day_x = 1

        self.end_train_x_list = [str(fold_id) for fold_id in setting['fold_id_list_csv'].split(',')]
        # self.end_train_default = self.end_train_x_list[0]

        for end_train_x in self.end_train_x_list:
            (self.result_dir_path / str(end_train_x)).mkdir(parents=True, exist_ok=True)
            (self.work_dir_path / str(end_train_x)).mkdir(parents=True, exist_ok=True)
            (self.model_dir_path / str(end_train_x)).mkdir(parents=True, exist_ok=True)

        # self.end_train_day_x = None

        # self.prediction_horizon_list = [int(prediction_horizon) for prediction_horizon in
        #                                 setting['prediction_horizon_list_csv'].split(',')]
        # self.prediction_horizon = None
        # self.prediction_horizon_prev = None

        # self.main_index_list = ['id', 'd']

        # self.remove_features = ['id', 'state_id', 'store_id', 'date', 'wm_yr_wk', 'd', self.target]
        # self.enable_features = None
        # self.mean_features = [
        #     'enc_cat_id_mean', 'enc_cat_id_std',
        #     'enc_dept_id_mean', 'enc_dept_id_std',
        #     'enc_item_id_mean', 'enc_item_id_std'
        # ]

        return