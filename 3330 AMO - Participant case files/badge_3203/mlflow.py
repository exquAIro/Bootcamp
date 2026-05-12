import logging
import subprocess
import sys
import mlflow
import optuna

from badge_3203.config import MLFLOW_HOST, MLFLOW_PORT


host = MLFLOW_HOST
port = str(MLFLOW_PORT)
tracking_uri = f"http://{host}:{port}"

def start_mlflow():
    cmd = [sys.executable, "-m", "mlflow", "server", "--host", host, "--port", port]
    process = subprocess.Popen(cmd)
    print(f"MLflow is running at: {tracking_uri}")
    return process



def stop_mlflow(process):
    process.terminate()


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def start_experiment(experiment_name):
    """
    Sets up the mlflow experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = get_or_create_experiment(experiment_name)

    mlflow.set_experiment(experiment_id=experiment_id)

    # override Optuna's default logging to ERROR only
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    return experiment_id


def log_search_parameters_to_mlflow(search):
    """
    Log search parameters to MLflow.
    """
    mlflow.log_params(search.best_params_)
    mlflow.log_param("cv_folds", search.cv if isinstance(search.cv, int) else search.cv.n_splits)
    param_key = "param_distributions" if hasattr(search, "param_distributions") else "param_grid"
    mlflow.log_param("search_params", getattr(search, param_key, None))


def log_cv_model_to_mlflow(params, cv_results, i):
    """
    Log common metrics (parameters, mean and std scores) to MLflow.
    """
    mlflow.log_params(params)

    mlflow.log_metric("mean_test_score", cv_results["mean_test_score"][i])
    mlflow.log_metric("std_test_score", cv_results["std_test_score"][i])
    mlflow.log_metric("mean_fit_time", cv_results["mean_fit_time"][i])
    mlflow.log_metric("std_fit_time", cv_results["std_fit_time"][i])

    if "mean_train_score" in cv_results:
        mlflow.log_metric("mean_train_score", cv_results["mean_train_score"][i])
    if "std_train_score" in cv_results:
        mlflow.log_metric("std_train_score", cv_results["std_train_score"][i])


def log_folds_to_mlflow(params, fold_scores, fold_train_scores=None):
    """
    Log cross-validation fold scores to MLflow.
    """
    for fold_index, fold_score in enumerate(fold_scores):
        with mlflow.start_run(run_name=f"Fold {fold_index + 1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("test_score", fold_score)
            if fold_train_scores and fold_index < len(fold_train_scores):
                mlflow.log_metric("train_score", fold_train_scores[fold_index])


def log_search_to_mlflow(search, run_name, log_cv=True, nested=False):
    """
    Log all CV runs of a SearchCV instance to MLflow,
    organizing trials and CV results hierarchically.
    """
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    cv_results = search.cv_results_

    # Optuna search objects don't add the params to the CV results
    if isinstance(search, optuna.integration.OptunaSearchCV):
        cv_results["params"] = [trial.params for trial in search.study.trials]

    with mlflow.start_run(run_name=run_name, nested=nested):
        log_search_parameters_to_mlflow(search)

        for i, params in enumerate(cv_results["params"]):

            with mlflow.start_run(run_name=f"Trial {i + 1}", nested=True):
                log_cv_model_to_mlflow(params, cv_results, i)

                if log_cv:
                    fold_scores = [
                        cv_results[f"split{fold_index}_test_score"][i]
                        for fold_index in range(search.cv.n_splits)
                    ]
                    fold_train_scores = [
                        cv_results.get(f"split{fold_index}_train_score", [None])[i]
                        for fold_index in range(search.cv.n_splits)
                    ]
                    log_folds_to_mlflow(params, fold_scores, fold_train_scores)


def log_cross_validation_to_mlflow(cv_results, run_name):
    """
    Log CV results and all underlying models to MLflow.
    """
    with mlflow.start_run(run_name=f"CV {run_name}"):
        mlflow.log_metric("mean_test_score", cv_results["test_score"].mean())
        mlflow.log_metric("std_test_score", cv_results["test_score"].std())
        mlflow.log_metric("mean_fit_time", cv_results["fit_time"].mean())
        mlflow.log_metric("std_fit_time", cv_results["fit_time"].std())

        if "train_score" in cv_results:
            mlflow.log_metric("mean_train_score", cv_results["train_score"].mean())
            mlflow.log_metric("std_train_score", cv_results["train_score"].std())

        for fold_index, estimator in enumerate(cv_results["estimator"]):
            log_search_to_mlflow(estimator, run_name=f"CV Fold {fold_index + 1}", nested=True)
