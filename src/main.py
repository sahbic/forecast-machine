import logging
import argparse
import datetime

import utils

# import m5a
# project_key = 'm5a'

from functions.tomahawk import *
project_key = 'tomahawk'

class myProgram(object):

    def __init__(self, params):
        self.params = params
        self.log = self.params.log
        return

    def main(self):
        self.log.info('main')
        result_dir_org_path = self.params.result_dir_path
        work_dir_org_path = self.params.work_dir_path
        model_dir_org_path = self.params.model_dir_path
        self.log.info('generate abt')
        generate_abt(self.params.raw_dir_path, work_dir_org_path)
        self.log.info('feature engineering')
        generate_features(self.params, self.log)
        self.log.info('split data')
        for end_train_x in self.params.end_train_x_list:
            self.params.end_train_x = end_train_x
            self.params.result_dir_path = result_dir_org_path / str(end_train_x)
            self.params.work_dir_path = work_dir_org_path / str(end_train_x)
            self.params.model_dir_path = model_dir_org_path / str(end_train_x)
            # self.params.update_file_path()

        
            for predict_horizon in self.params.prediction_horizon_list:
                self.log.info('-----------------', 'fold_id', end_train_x, 'predict_horizon', predict_horizon)

        return

def main():
    # parser
    parser = argparse.ArgumentParser(description="Forecasting script")
    # Set the default arguments
    parser.add_argument('-ddp', '--data_dir_path', type=str, default='.')
    parser.add_argument('-opn', '--output_name', type=str, default='default')
    parser.add_argument('-t', '--time_column', type=str, default='date')
    parser.add_argument('-npr', '--number_predictions', type=str, default=6)
    parser.add_argument('-npg', '--n_predictions_groupby', type=str, default=6)
    parser.add_argument('-flc', '--fold_id_list_csv', type=str, default='2017-06')

    args = parser.parse_args()

    params = utils.Params(
    {
        'data_dir_path': args.data_dir_path,
        'output_name': args.output_name,
        'log_name': '{}_main'.format(project_key),
        'project_key': project_key,
        'time_col': args.time_column,
        'number_predictions': args.number_predictions,
        'n_predictions_groupby': args.n_predictions_groupby,
        'fold_id_list_csv': args.fold_id_list_csv
    })

    m = myProgram(params)
    m.log.info('******** start')
    m.log.info(parser.parse_args())
    m.start_dt = datetime.datetime.now()

    m.log.info(m.params.setting)
    m.main()
    m.log.info(['******** end', 'start_time', m.start_dt, 'process_time', datetime.datetime.now() - m.start_dt])


if __name__ == '__main__':
    main()