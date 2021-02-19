import logging
import argparse
import datetime
import pickle

import pandas as pd
import numpy as np

import utils
import funcs
from models import Mean, MeanTS, Last, RandomForest


class myProgram(object):

    def __init__(self, params):
        self.params = params
        self.log = self.params.log
        return

    def main(self):
        custom = __import__("functions."+self.params.project_key, fromlist=['generate_abt', 'generate_features', 'get_offset'])

        self.log.info('main')
        result_dir_org_path = self.params.result_dir_path
        work_dir_org_path = self.params.work_dir_path
        model_dir_org_path = self.params.model_dir_path
        self.log.info('generate abt')
        df = custom.generate_abt(self.params.raw_dir_path, work_dir_org_path)
        self.log.info('feature engineering')
        df = custom.generate_features(df, self.params, self.log)

        if self.params.segment_groupby_column:
            segments_list = df[self.params.segment_groupby_column].unique()
        else:
            segments_list = ["all"]

        scores = []
        # iterate over folds
        for end_train_x in self.params.end_train_x_list:
            self.params.end_train_x = end_train_x
            self.params.result_dir_path = result_dir_org_path / str(end_train_x)
            self.params.work_dir_path = work_dir_org_path / str(end_train_x)
            self.params.model_dir_path = model_dir_org_path / str(end_train_x)
            # self.params.update_file_path()
            end_train_time = pd.to_datetime(end_train_x)
            offset = custom.get_offset(self.params.number_predictions)
            end_test_time = end_train_time + offset
            self.log.info('train end date {} / test date end {}'.format(str(end_train_time), str(end_test_time)))
            train_fold, test_fold = funcs.split(df, self.params.time_col, end_train_time, end_test_time)
        
            res_tgroups = pd.DataFrame()
            # iterate time groups ,each time group has a different prediction horizon
            for predict_horizon in self.params.prediction_horizon_list:
                self.log.info('-----------------', 'fold_id', end_train_x, 'predict_horizon', predict_horizon)
                begin_test_time_group = end_train_time + custom.get_offset(predict_horizon - self.params.n_predictions_groupby + 1) 
                end_test_time_group = end_train_time + custom.get_offset(predict_horizon)
                self.log.info('test begin', str(begin_test_time_group), 'test end', str(end_test_time_group))

                self.log.info('shift data')
                # shifted = funcs.shift_with_pred_horizon(df, self.params.dependent_var, predict_horizon)
                shifted = funcs.add_last_value(df, self.params.id_col, self.params.dependent_var, predict_horizon)

                res_segments = pd.DataFrame()
                # iterate over segments
                for segment in segments_list:
                    # split
                    # self.log.info('split data')
                    if segment == "all":
                        part_df = shifted.copy()
                        self.log.info('data shape '+ str(part_df.shape))
                    else:
                        self.log.info('--------', 'segment', segment)
                        segmented = shifted[shifted[self.params.segment_groupby_column] == segment]
                        part_df = segmented.copy()
                        self.log.info('segment shape '+ str(part_df.shape))

                    part_df = funcs.drop_constant_columns(part_df, self.log)
                    train, test = funcs.split_with_time_grouping(part_df, self.params.time_col, end_train_time, begin_test_time_group, end_test_time_group)

                    estimator = RandomForest(self.params.id_col, self.params.time_col, self.params.dependent_var, self.log)
                    estimator.fit(train)
                    res = estimator.predict(test)
                    # collect results of segments
                    res_segments = pd.concat([res_segments, res])
                
                # collect results of time groups
                res_tgroups = pd.concat([res_tgroups, res_segments])

            # compute fold error
            self.log.info('test result shape'+ str(res_tgroups.shape))
            error = funcs.compute_metric(res_tgroups, test_fold, self.params)
            self.log.info('fold_id', end_train_x, 'error', error)
            # compute fold results
            scores.append(error)

        cross_fold_error = np.mean(scores)
        self.log.info('=> cross fold mean error', cross_fold_error)


        return

def main():    
    # parser
    parser = argparse.ArgumentParser(description="Forecasting script")
    # Set the default arguments
    parser.add_argument('-pr', '--project_key', type=str, default='m5a')
    parser.add_argument('-ddp', '--data_dir_path', type=str, default='.')
    parser.add_argument('-opn', '--output_name', type=str, default='default')
    parser.add_argument('-t', '--time_column', type=str, default='date')
    parser.add_argument('-id', '--id_column', type=str, default='id')
    parser.add_argument('-dv', '--dependent_var', type=str, default='value')
    parser.add_argument('-npr', '--number_predictions', type=str, default="6")
    parser.add_argument('-npg', '--n_predictions_groupby', type=str, default="6")
    parser.add_argument('-sva', '--segment_groupby_column', type=str)
    parser.add_argument('-flc', '--fold_id_list_csv', type=str, default='2017-06')

    args = parser.parse_args()

    params = utils.Params(
    {
        'data_dir_path': args.data_dir_path,
        'output_name': args.output_name,
        'log_name': '{}_main'.format(args.project_key),
        'project_key': args.project_key,
        'id_col': args.id_column,
        'time_col': args.time_column,
        'dependent_var': args.dependent_var,
        'number_predictions': args.number_predictions,
        'n_predictions_groupby': args.n_predictions_groupby,
        'segment_groupby_column': args.segment_groupby_column,
        'fold_id_list_csv': args.fold_id_list_csv,
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