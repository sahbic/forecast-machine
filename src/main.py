import json
import pickle

import mlflow
from mlflow.entities import ViewType

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
    # TODO: use yaml config file for each project
    models = {
        "ph_models": [
            RandomForest(
                id_col, time_col, dependent_var, log
            ),
            Last(id_col, time_col, dependent_var, log),
        ]
    }
    return models

def delete_experiment(experiment_name: str):
    """Delete an experiment with name `experiment_name`.
    Args:
        experiment_name (str): Name of the experiment.
    """
    mlflow_client = mlflow.tracking.MlflowClient()
    experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id
    mlflow_client.delete_experiment(experiment_id=experiment_id)

# TODO:
# - what to do if no input ?
# - add the number of iterations to yaml config file
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
    stores_dir,
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

    # mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_tracking_uri(stores_dir)
    tracking_uri = mlflow.get_tracking_uri()
    log.info("Current tracking uri: {}".format(tracking_uri))
    mlflow_client = mlflow.tracking.MlflowClient()

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
            
            # Start experiment
            model_name = project_key + "_" + segment + "_" + str(predict_horizon)
            mlflow.set_experiment(model_name)

            experiment_id = mlflow.get_experiment_by_name(model_name).experiment_id
            log.info("experiment id {}".format(experiment_id))

            with mlflow.start_run() as parent_run:

                for model in models["ph_models"]:
                    log.info("training " + model.name)

                    tags = {
                        "segment" : segment,
                        "prediction_horizon": predict_horizon,
                        "model_name": model.name,
                        "n_grid_features": grid_ph_seg.shape[1]
                    }

                    model.tune_fit(grid_ph_seg, tscv, 2)
                    model.track(experiment_id, tags, n_folds)

            # Get best run
            df = mlflow.search_runs(
                experiment_ids=experiment_id,
                filter_string="tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id),
                order_by=["metric.average_mse"],
                output_format="pandas",
            )

            run_id = df.loc[df['metrics.average_mse'].idxmin()]['run_id']
            results.append(mlflow.get_run(run_id).to_dictionary())
            scores.append(mlflow.get_run(run_id).data.metrics["average_mse"])

            try:
                mlflow_client.delete_registered_model(name=model_name)
            except mlflow.exceptions.MlflowException:
                pass

            result = mlflow.register_model(
                    "runs:/{}/model".format(run_id),
                    model_name
                )

            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=1,
                stage="Production"
            )

    # Mean of all test error means
    mean_error_means = np.mean(scores)
    # save results
    result_path = stores_dir / "final_models.json"
    log.info("Save final model details to " + str(result_path))
    with open(result_path, "w") as outfile:
        json.dump(results, outfile)

    # Start experiment
    mlflow.set_experiment(project_key)
    with mlflow.start_run():
        # Track input file
        mlflow.log_artifact(f"{work_dir_path / input_file_name}")

        # Track parameters    
        parameters = {
            "number_predictions" : number_predictions,
            "n_predictions_groupby" : n_predictions_groupby,
            "column_segment_groupby" : segment_groupby_column,
            "n_folds" : n_folds
        }
        mlflow.log_params(parameters)

        # Track column names
        tags = {
            "id_column" : id_col,
            "time_column" : time_col,
            "target" : dependent_var
        }
        mlflow.set_tags(tags) 

        # Track metrics
        mlflow.log_metric("average_cv_mse", mean_error_means)

        # Track models selected
        mlflow.log_artifact(result_path)

    # Mean of all test error means
    mean_error_means = np.mean(scores)
    log.info(f"Mean of test error means: {mean_error_means:.2f}")

    return

# TODO:
# - s'assurer que les modèles enregistrés correspondent bien au best run du projet ? comment ?
#       ou bien utiliser le best run du projet pour aller récupérer runs et modèles associés
# - what to do if no train ?
def backtest(
    project_key,
    run_id,
    n_folds,
    work_dir_path,
    input_file_name,
    log
):
    custom = __import__(
        project_key + ".prepare", fromlist=["generate_grid", "get_offset"]
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow_client = mlflow.tracking.MlflowClient()
    experiment_id = mlflow_client.get_experiment_by_name(project_key).experiment_id

    if not run_id:
    # get best run
        df = mlflow.search_runs(
            experiment_ids=experiment_id,
            order_by=["metric.average_cv_mse"],
            output_format="pandas",
        )
        run_id = df.loc[df['metrics.average_cv_mse'].idxmin()]['run_id']

    id_col = mlflow.get_run(run_id).data.tags["id_column"]
    time_col = mlflow.get_run(run_id).data.tags["time_column"]
    dependent_var = mlflow.get_run(run_id).data.tags["target"]
    number_predictions = int(mlflow.get_run(run_id).data.params["number_predictions"])
    n_predictions_groupby = int(mlflow.get_run(run_id).data.params["n_predictions_groupby"])
    segment_groupby_column = mlflow.get_run(run_id).data.params["column_segment_groupby"]

    base = load_base(work_dir_path, input_file_name, log)
    base[time_col] = pd.to_datetime(base[time_col])
    prediction_horizon_list = get_prediction_horizon_list(number_predictions, n_predictions_groupby)
    segments_list = get_segments_list(base, segment_groupby_column)

    splitter = funcs.get_splitter(base, time_col, n_folds, number_predictions)


    fold_scores_train, fold_scores_test = [], []
    test_set, train_set = pd.DataFrame(), pd.DataFrame()

    fold_idx = 0

    for train_indexes, test_indexes in splitter:
        fold_idx += 1
        log.info("------------------------ " + "fold_id " +  str(fold_idx))
        end_train_time = base.iloc[train_indexes][time_col].max()
        end_test_time = base.iloc[test_indexes][time_col].max()
        log.info(
            "train end date {} / test date end {}".format(
                str(end_train_time), str(end_test_time)
            )
        )

        # concat test sets for later evaluation
        test_set = pd.concat([test_set, base.iloc[test_indexes]])
        train_set = pd.concat([train_set, base.iloc[train_indexes]])

        res_tgroups_test, res_tgroups_train = pd.DataFrame(), pd.DataFrame()

        # iterate time groups ,each time group has a different prediction horizon
        for predict_horizon in prediction_horizon_list:
            log.info("---------------- " + "predict_horizon " + str(predict_horizon))

            # get time boundaries
            begin_test_time_group = end_train_time + custom.get_offset(
                predict_horizon - n_predictions_groupby + 1
            )
            end_test_time_group = end_train_time + custom.get_offset(predict_horizon)
            log.info(
                "valid_ph begin " +
                str(begin_test_time_group) +
                " valid_ph end " +
                str(end_test_time_group),
            )

            # generate grid, add temporal features with prediction horizon
            log.info("generate grid table")
            grid_ph = custom.generate_grid(base, id_col, dependent_var, predict_horizon, work_dir_path, log)

            res_segments_test, res_segments_train = pd.DataFrame(), pd.DataFrame()

            # iterate over segments
            for segment in segments_list:
                if segment == "all":
                    grid_ph_seg = grid_ph.copy()
                    log.info("grid_ph " + str(grid_ph_seg.shape))
                else:
                    log.info("-------- " + "segment " + segment)
                    segmented = grid_ph[grid_ph[segment_groupby_column] == segment]
                    grid_ph_seg = segmented.copy()
                    del segmented
                    log.info("grid_ph_seg " + str(grid_ph_seg.shape))

                # get train test sets
                train, test = funcs.split_with_time_grouping(
                    grid_ph_seg,
                    time_col,
                    end_train_time,
                    begin_test_time_group,
                    end_test_time_group,
                )

                # get model (the object not the trained model because retrained on all dataset)
                model_name = project_key + "_" + segment + "_" + str(predict_horizon)
                stage = 'Production'

                run_id = mlflow_client.get_latest_versions(model_name, stages=[stage])[0].run_id

                model_path = mlflow.get_run(run_id).info.artifact_uri + "/model/python_model.pkl"
                model = pickle.load(open(model_path, "rb"))

                model.fit_with_params(train)

                log.info("{} {}".format(model.name,model.best_params))
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
        test_error = funcs.compute_metric(res_tgroups_test, test_set, id_col, time_col, dependent_var)
        train_error = funcs.compute_metric(res_tgroups_train, train_set, id_col, time_col, dependent_var)
        # compute fold results
        fold_scores_test.append(test_error)
        fold_scores_train.append(train_error)

    log.info(
        "=> train fold errors " + str(["{:.3f}".format(error) for error in fold_scores_train])
    )
    log.info("=> train mean error " + "{:.6f}".format(np.mean(fold_scores_train)))
    log.info("=> test fold errors " + str(["{:.3f}".format(error) for error in fold_scores_test]))
    log.info("=> test mean error " + "{:.6f}".format(np.mean(fold_scores_test)))

    return