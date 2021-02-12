import logging
import argparse
import datetime
import pickle

import utils
import funcs
import algs

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
        df = generate_abt(self.params.raw_dir_path, work_dir_org_path)
        self.log.info('feature engineering')
        df = generate_features(df, self.params, self.log)

        if self.params.segment_groupby_column:
            segments_list = df[self.params.segment_groupby_column].unique()
        else:
            segments_list = ["all"]

        for end_train_x in self.params.end_train_x_list:
            self.params.end_train_x = end_train_x
            self.params.result_dir_path = result_dir_org_path / str(end_train_x)
            self.params.work_dir_path = work_dir_org_path / str(end_train_x)
            self.params.model_dir_path = model_dir_org_path / str(end_train_x)
            # self.params.update_file_path()
            end_train_time = pd.to_datetime(end_train_x)
            offset = get_offset(self.params.number_predictions)
            end_validation_time = end_train_time + offset
            self.log.info('train end date {} / validate date end {}'.format(str(end_train_time), str(end_validation_time)))
        
            for predict_horizon in self.params.prediction_horizon_list:
                self.log.info('-----------------', 'fold_id', end_train_x, 'predict_horizon', predict_horizon)
                self.log.info('shift data')
                shifted = funcs.shift_with_pred_horizon(df, self.params.dependent_var, predict_horizon)

                for segment in segments_list:
                    # split
                    self.log.info('split data')
                    if segment == "all":
                        full_df = shifted.copy()
                    else:
                        self.log.info('--------', 'segment', segment)
                        segmented = shifted[shifted[self.params.segment_groupby_column] == segment]
                        full_df = segmented.copy()

                    num_cols, cat_cols = funcs.get_num_cat_columns(df, self.params, self.log)
                    full_df_rf = funcs.preprocess_ml_sklearn_forests(full_df, cat_cols, self.params, self.log)
                    train_rf, validate_rf = funcs.split(full_df_rf, self.params.time_col, end_train_time, end_validation_time)
                    train_rf, validate_rf = funcs.impute_missing_mean(train_rf, validate_rf, num_cols, self.params, self.log)
                    # train
                    self.log.info('train')
                    model_name = str(self.params.model_dir_path / f'lgb_model_{segment}_{predict_horizon}.bin')
                    estimator = algs.train_rf_model(train_rf, self.params, self.log)
                    pickle.dump(estimator, open(model_name, 'wb'))
                    # evaluate
                    # res = algs.predict_rf_model(validate_rf, self.params, self.log)
                
                # temp_path = str(self.params.work_dir_path / "temp.csv")
                # shifted.to_csv(temp_path, index=False)


        return

def main():
    # parser
    parser = argparse.ArgumentParser(description="Forecasting script")
    # Set the default arguments
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
        'log_name': '{}_main'.format(project_key),
        'project_key': project_key,
        'id_col': args.id_column,
        'time_col': args.time_column,
        'dependent_var': args.dependent_var,
        'number_predictions': args.number_predictions,
        'n_predictions_groupby': args.n_predictions_groupby,
        'segment_groupby_column': args.segment_groupby_column,
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