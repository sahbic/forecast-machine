import argparse
import datetime
import json
import logging
import pickle

import numpy as np
import pandas as pd

import funcs
import utils
from models import Last, Mean, MeanTS, RandomForest


class myProgram(object):
    def __init__(self, params):
        self.params = params
        self.log = self.params.log
        return

    def main(self):
        custom = __import__("functions." + self.params.project_key, fromlist=["generate_base"])

        self.log.info("main")

        self.log.info("generate base table")
        base = custom.generate_base(
            self.params.raw_dir_path, self.params.work_dir_path, self.params, self.log
        )

        # get segments_list
        if self.params.segment_groupby_column:
            self.params.segments_list = base[self.params.segment_groupby_column].unique()
        else:
            self.params.segments_list = ["all"]

        models = {
            "ph_models": [
                RandomForest(
                    self.params.id_col, self.params.time_col, self.params.dependent_var, self.log
                ),
                Last(self.params.id_col, self.params.time_col, self.params.dependent_var, self.log),
            ]
        }

        model = RandomForest(
            self.params.id_col, self.params.time_col, self.params.dependent_var, self.log
        )

        if self.params.mode == "train_eval":
            self.train_eval(base, model)
        if self.params.mode == "tune_train_eval":
            self.tune_train_eval(base, models)
        if self.params.mode == "backtest":
            self.backtest(base)

    def tune_train_eval(self, base, models):
        custom = __import__(
            "functions." + self.params.project_key, fromlist=["generate_grid", "get_offset"]
        )

        results = []
        scores = []

        # iterate over time groups ,each time group has a different prediction horizon
        for predict_horizon in self.params.prediction_horizon_list:
            self.log.info("----------------", "predict_horizon", predict_horizon)

            # generate grid, add temporal features with prediction horizon
            self.log.info("generate grid table")
            grid_ph = custom.generate_grid(base, self.params, self.log, predict_horizon)

            # iterate over segments
            for segment in self.params.segments_list:
                if segment == "all":
                    grid_ph_seg = grid_ph.copy()
                    self.log.info("grid_ph " + str(grid_ph_seg.shape))
                else:
                    self.log.info("--------", "segment", segment)
                    segmented = grid_ph[grid_ph[self.params.segment_groupby_column] == segment]
                    grid_ph_seg = segmented.copy()
                    del segmented
                    self.log.info("grid_ph_seg " + str(grid_ph_seg.shape))

                # get CV indexes
                grid_ph_seg = grid_ph_seg.reset_index()
                tscv = funcs.get_splitter(grid_ph_seg, self.params)

                model_scores = []
                model_results = []
                for model in models["ph_models"]:
                    result = {}
                    # fit predict
                    model.tune_fit(grid_ph_seg, tscv, 1)

                    model_scores.append(model.cv_results["mean_test_score"][model.best_index])

                    # document results
                    result["segment"] = segment
                    result["prediction_horizon"] = predict_horizon
                    result["model_name"] = model.name
                    result["mean_test_score"] = model.cv_results["mean_test_score"][
                        model.best_index
                    ]
                    scores.append(result["mean_test_score"])
                    for i in range(self.params.n_folds):
                        result["split" + str(i) + "_test_score"] = model.cv_results[
                            "split" + str(i) + "_test_score"
                        ][model.best_index]
                    result["params"] = model.best_params
                    model_results.append(result)

                # best model has smallest MSE
                index_best = np.argmin(model_scores)
                best_model = models["ph_models"][index_best]
                results.append(model_results[index_best])
                # save model
                pickle_name = "model" + "_" + segment + "_" + str(predict_horizon) + ".pkl"
                model_path = self.params.model_dir_path / pickle_name
                self.log.info(
                    "Save model for segment {} prediction horizon {}".format(
                        segment, str(predict_horizon)
                    )
                )
                pickle.dump(best_model, open(model_path, "wb"))

        # Mean of all test error means
        mean_error_means = np.mean(scores)
        self.log.info("Mean of test error means", mean_error_means)
        # save results
        result_path = self.params.result_dir_path / "final_models.json"
        self.log.info("Save final model details to", result_path)
        with open(result_path, "w") as outfile:
            json.dump(results, outfile)

        return

    def backtest(self, base):
        custom = __import__(
            "functions." + self.params.project_key, fromlist=["generate_grid", "get_offset"]
        )

        base = base.reset_index()
        splitter = funcs.get_splitter(base, self.params)

        fold_scores_train, fold_scores_test = [], []
        test_set, train_set = pd.DataFrame(), pd.DataFrame()

        fold_idx = 0

        for train_indexes, test_indexes in splitter:
            fold_idx += 1
            self.log.info("------------------------", "fold_id", fold_idx)
            end_train_time = base.iloc[train_indexes][self.params.time_col].max()
            end_test_time = base.iloc[test_indexes][self.params.time_col].max()
            self.log.info(
                "train end date {} / test date end {}".format(
                    str(end_train_time), str(end_test_time)
                )
            )

            # concat test sets for later evaluation
            test_set = pd.concat([test_set, base.iloc[test_indexes]])
            train_set = pd.concat([train_set, base.iloc[train_indexes]])

            res_tgroups_test, res_tgroups_train = pd.DataFrame(), pd.DataFrame()

            # iterate time groups ,each time group has a different prediction horizon
            for predict_horizon in self.params.prediction_horizon_list:
                self.log.info("----------------", "predict_horizon", predict_horizon)

                # get time boundaries
                begin_test_time_group = end_train_time + custom.get_offset(
                    predict_horizon - self.params.n_predictions_groupby + 1
                )
                end_test_time_group = end_train_time + custom.get_offset(predict_horizon)
                self.log.info(
                    "valid_ph begin",
                    str(begin_test_time_group),
                    "valid_ph end",
                    str(end_test_time_group),
                )

                # generate grid, add temporal features with prediction horizon
                self.log.info("generate grid table")
                grid_ph = custom.generate_grid(base, self.params, self.log, predict_horizon)

                res_segments_test, res_segments_train = pd.DataFrame(), pd.DataFrame()

                # iterate over segments
                for segment in self.params.segments_list:
                    if segment == "all":
                        grid_ph_seg = grid_ph.copy()
                        self.log.info("grid_ph " + str(grid_ph_seg.shape))
                    else:
                        self.log.info("--------", "segment", segment)
                        segmented = grid_ph[grid_ph[self.params.segment_groupby_column] == segment]
                        grid_ph_seg = segmented.copy()
                        del segmented
                        self.log.info("grid_ph_seg " + str(grid_ph_seg.shape))

                    # get train test sets
                    train, test = funcs.split_with_time_grouping(
                        grid_ph_seg,
                        self.params.time_col,
                        end_train_time,
                        begin_test_time_group,
                        end_test_time_group,
                    )

                    # get model (the object not the trained model because retrained on all dataset)
                    pickle_name = "model" + "_" + segment + "_" + str(predict_horizon) + ".pkl"
                    model_path = self.params.model_dir_path / pickle_name
                    model = pickle.load(open(model_path, "rb"))
                    self.log.info(model.name, model.best_params)
                    # fit predict
                    model.fit_with_params(train)

                    res_test = model.predict(test)
                    res_train = model.predict(train)

                    # collect results of segments

                    res_segments_test = pd.concat([res_segments_test, res_test])
                    res_segments_train = pd.concat([res_segments_train, res_train])

                # collect results of time groups
                res_tgroups_test = pd.concat([res_tgroups_test, res_segments_test])
                res_tgroups_train = pd.concat([res_tgroups_train, res_segments_train])

            # compute fold error
            test_error = funcs.compute_metric(res_tgroups_test, test_set, self.params)
            train_error = funcs.compute_metric(res_tgroups_train, train_set, self.params)
            # compute fold results
            fold_scores_test.append(test_error)
            fold_scores_train.append(train_error)

        self.log.info(
            "=> train fold errors", ["{:.3f}".format(error) for error in fold_scores_train]
        )
        self.log.info("=> train mean error", "{:.6f}".format(np.mean(fold_scores_train)))
        self.log.info("=> test fold errors", ["{:.3f}".format(error) for error in fold_scores_test])
        self.log.info("=> test mean error", "{:.6f}".format(np.mean(fold_scores_test)))

        return

    def train_eval(self, base, model):
        custom = __import__(
            "functions." + self.params.project_key, fromlist=["generate_grid", "get_offset"]
        )

        fold_scores_train, fold_scores_test = [], []

        # iterate over folds
        for end_train_x in self.params.end_train_x_list:
            self.log.info("------------------------", "fold_id", end_train_x)

            # get time boundaries
            self.params.end_train_x = end_train_x
            end_train_time = pd.to_datetime(end_train_x)
            offset = custom.get_offset(self.params.number_predictions)
            end_test_time = end_train_time + offset
            self.log.info(
                "train end date {} / test date end {}".format(
                    str(end_train_time), str(end_test_time)
                )
            )
            # get test set for later evaluation
            train_fold, test_fold = funcs.split(
                base, self.params.time_col, end_train_time, end_test_time
            )

            res_tgroups_test, res_tgroups_train = pd.DataFrame(), pd.DataFrame()

            # iterate time groups ,each time group has a different prediction horizon
            for predict_horizon in self.params.prediction_horizon_list:
                self.log.info("----------------", "predict_horizon", predict_horizon)

                # get time boundaries
                begin_test_time_group = end_train_time + custom.get_offset(
                    predict_horizon - self.params.n_predictions_groupby + 1
                )
                end_test_time_group = end_train_time + custom.get_offset(predict_horizon)
                self.log.info(
                    "valid_ph begin",
                    str(begin_test_time_group),
                    "valid_ph end",
                    str(end_test_time_group),
                )

                # generate grid, add temporal features with prediction horizon
                self.log.info("generate grid table")
                grid_ph = custom.generate_grid(base, self.params, self.log, predict_horizon)

                res_segments_test, res_segments_train = pd.DataFrame(), pd.DataFrame()

                # iterate over segments
                for segment in self.params.segments_list:
                    if segment == "all":
                        grid_ph_seg = grid_ph.copy()
                        self.log.info("grid_ph " + str(grid_ph_seg.shape))
                    else:
                        self.log.info("--------", "segment", segment)
                        segmented = grid_ph[grid_ph[self.params.segment_groupby_column] == segment]
                        grid_ph_seg = segmented.copy()
                        del segmented
                        self.log.info("grid_ph_seg " + str(grid_ph_seg.shape))

                    # get train test sets
                    train, test = funcs.split_with_time_grouping(
                        grid_ph_seg,
                        self.params.time_col,
                        end_train_time,
                        begin_test_time_group,
                        end_test_time_group,
                    )

                    # fit predict
                    model.fit(train)

                    res_test = model.predict(test)
                    res_train = model.predict(train)

                    # collect results of segments
                    res_segments_test = pd.concat([res_segments_test, res_test])
                    res_segments_train = pd.concat([res_segments_train, res_train])

                # collect results of time groups
                res_tgroups_test = pd.concat([res_tgroups_test, res_segments_test])
                res_tgroups_train = pd.concat([res_tgroups_train, res_segments_train])

            # compute fold error
            test_error = funcs.compute_metric(res_tgroups_test, test_fold, self.params)
            train_error = funcs.compute_metric(res_tgroups_train, train_fold, self.params)
            # compute fold results
            fold_scores_test.append(test_error)
            fold_scores_train.append(train_error)

        self.log.info(
            "=> train fold errors", ["{:.3f}".format(error) for error in fold_scores_train]
        )
        self.log.info("=> train mean error", "{:.6f}".format(np.mean(fold_scores_train)))
        self.log.info("=> test fold errors", ["{:.3f}".format(error) for error in fold_scores_test])
        self.log.info("=> test mean error", "{:.6f}".format(np.mean(fold_scores_test)))

        return


def main():
    # parser
    parser = argparse.ArgumentParser(description="Forecasting script")
    # Set the default arguments
    parser.add_argument("-pr", "--project_key", type=str, default="m5a")
    parser.add_argument("-ddp", "--data_dir_path", type=str, default=".")
    parser.add_argument("-opn", "--output_name", type=str, default="default")
    parser.add_argument("-t", "--time_column", type=str, default="date")
    parser.add_argument("-id", "--id_column", type=str, default="id")
    parser.add_argument("-dv", "--dependent_var", type=str, default="value")
    parser.add_argument("-npr", "--number_predictions", type=str, default="6")
    parser.add_argument("-npg", "--n_predictions_groupby", type=str, default="6")
    parser.add_argument("-sva", "--segment_groupby_column", type=str)
    parser.add_argument("-flc", "--fold_id_list_csv", type=str, default="2017-06")
    parser.add_argument("-mod", "--mode", type=str, default="train_eval")
    parser.add_argument("-nfo", "--n_folds", type=str, default="3")

    args = parser.parse_args()

    params = utils.Params(
        {
            "data_dir_path": args.data_dir_path,
            "output_name": args.output_name,
            "log_name": "{}_main".format(args.project_key),
            "project_key": args.project_key,
            "id_col": args.id_column,
            "time_col": args.time_column,
            "dependent_var": args.dependent_var,
            "number_predictions": args.number_predictions,
            "n_predictions_groupby": args.n_predictions_groupby,
            "segment_groupby_column": args.segment_groupby_column,
            "fold_id_list_csv": args.fold_id_list_csv,
            "mode": args.mode,
            "n_folds": args.n_folds,
        }
    )

    m = myProgram(params)
    m.log.info("******** start")
    m.log.info(parser.parse_args())
    m.start_dt = datetime.datetime.now()

    m.log.info(m.params.setting)
    m.main()
    m.log.info(
        [
            "******** end",
            "start_time",
            m.start_dt,
            "process_time",
            datetime.datetime.now() - m.start_dt,
        ]
    )


if __name__ == "__main__":
    main()
