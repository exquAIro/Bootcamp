import pandas as pd
from IPython.display import display
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(y_true, y_pred):
    """
    Helper function to calculate common regression metrics.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - metrics: Dictionary containing evaluation metrics
    """

    metrics = {
        "r2_score": r2_score(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_true, y_pred),
    }

    return metrics


def evaluate_model(model, X_train, y_train, X_test, y_test, verbose=True):
    """
    Evaluates a fitted regression model on both training and test data
    and prints common performance metrics for each if verbose=True.

    Parameters:
    - model: Fitted model to evaluate
    - X_train: Features of the training dataset
    - y_train: True labels of the training dataset
    - X_test: Features of the test dataset
    - y_test: True labels of the test dataset
    - verbose: Whether to display the metrics. Default is True.

    Returns:
    - train_metrics: Dictionary containing evaluation metrics for the training set
    - test_metrics: Dictionary containing evaluation metrics for the test set
    """

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    combined_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
    combined_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    if verbose:
        # Convert metrics to DataFrame for better display
        metrics_df = pd.DataFrame(
            {
                "Metric": list(train_metrics.keys()),
                "Training Set": list(train_metrics.values()),
                "Test Set": list(test_metrics.values()),
            }
        )

        display(name="Model Evaluation Metrics", dataframe=metrics_df)

    return combined_metrics
