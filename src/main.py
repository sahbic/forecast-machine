import json
import pickle

import numpy as np
import pandas as pd

from src import funcs
from src.models import Last, Mean, MeanTS, RandomForest


def load_base(work_dir_path, input_file_name, log):
    log.info("get base table")
    file_path = work_dir_path / input_file_name
    base = pd.read_csv(file_path)
    return base

def get_segments_list(base, segment_groupby_column):
    if segment_groupby_column:
        segments_list = base[segment_groupby_column].unique()
    else:
        segments_list = ["all"]
    return segments_list

def get_prediction_horizon_list(number_predictions, n_predictions_groupby):
    ratio_prediction = int(number_predictions / n_predictions_groupby)
    prediction_horizon_list = (
        [i * n_predictions_groupby for i in range(1, ratio_prediction + 1)]
        if (ratio_prediction > 1)
        else [number_predictions]
    )
    return prediction_horizon_list

def get_models(id_col, time_col, dependent_var, log):
    models = {
        "ph_models": [
            RandomForest(
                id_col, time_col, dependent_var, log
            ),
            Last(id_col, time_col, dependent_var, log),
        ]
    }
    return models

def train(
    project_key,
    id_col,
    time_col,
    dependent_var,
    number_predictions,
    n_predictions_groupby,
    segment_groupby_column,
    n_folds,
    work_dir_path,
    input_file_name,
    output_name,
    log
):
    custom = __import__(
        project_key + ".prepare", fromlist=["generate_grid", "get_offset"]
    )

    base = load_base(work_dir_path, input_file_name, log)
    base[time_col] = pd.to_datetime(base[time_col])
    prediction_horizon_list = get_prediction_horizon_list(number_predictions, n_predictions_groupby)
    segments_list = get_segments_list(base, segment_groupby_column)
    models = get_models(id_col, time_col, dependent_var, log)

    results = []
    scores = []

    # iterate over time groups ,each time group has a different prediction horizon
    for predict_horizon in prediction_horizon_list:
        log.info("---------------- " + "predict_horizon " + str(predict_horizon))

        # generate grid, add temporal features with prediction horizon
        log.info("generate grid table")
        grid_ph = custom.generate_grid(base, id_col, dependent_var, predict_horizon, work_dir_path, log)

        # iterate over segments
        for segment in segments_list:
            if segment == "all":
                grid_ph_seg = grid_ph.copy()
                log.info("train dataset of size " + str(grid_ph_seg.shape))
            else:
                log.info("-------- " + "segment " + segment)
                segmented = grid_ph[grid_ph[segment_groupby_column] == segment]
                grid_ph_seg = segmented.copy()
                del segmented
                log.info("train dataset of size " + str(grid_ph_seg.shape))

            # get CV indexes
            grid_ph_seg = grid_ph_seg.reset_index()
            tscv = funcs.get_splitter(grid_ph_seg, time_col, n_folds, number_predictions)

            model_scores = []
            model_results = []
            for model in models["ph_models"]:
                log.info("training " + model.name)
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
                for i in range(n_folds):
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
            # model_path = self.params.model_dir_path / pickle_name
            # log.info(
            #     "Save model for segment {} prediction horizon {}".format(
            #         segment, str(predict_horizon)
            #     )
            # )
            # pickle.dump(best_model, open(model_path, "wb"))

    # Mean of all test error means
    mean_error_means = np.mean(scores)
    log.info(f"Mean of test error means: {mean_error_means:.2f}")
    # save results
    # result_path = self.params.result_dir_path / "final_models.json"
    # log.info("Save final model details to " + result_path)
    # with open(result_path, "w") as outfile:
    #     json.dump(results, outfile)

    return