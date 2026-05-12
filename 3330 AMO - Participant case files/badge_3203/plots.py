import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, validation_curve


def plot_distributions(y_train, y_test, title="Distribution Plot", ax=None):
    # Create combined DataFrame for train and test sets
    combined_df = pd.DataFrame(
        {
            "Target": np.concatenate([y_train, y_test]),
            "Set": ["Train"] * len(y_train) + ["Test"] * len(y_test),
        }
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    sns.histplot(
        data=combined_df, x="Target", hue="Set", multiple="dodge", ax=ax, bins=30, kde=True
    )
    ax.set_title(title)
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")


def plot_correlation_with_demand(df, save_path=None):
    """
    Plots the correlation of each variable in the dataframe with the 'demand' column.

    Args:
    - df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
    - save_path (str, optional): Path to save the generated plot.
        If not specified, plot won't be saved.

    Returns:
    - None (Displays the plot on a Jupyter window)
    """

    # Compute correlations between all variables and 'demand'
    correlations = df.corr()["demand"].drop("demand").sort_values()

    # Generate a color palette from red to green
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )  # Light grey background and thicker grid lines

    # Create bar plot
    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Set labels and title with increased font size
    plt.title("Correlation with Demand", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # prevent matplotlib from displaying the chart every time we call this function
    plt.close(fig)

    return fig


# Example of function to plot the grid search results
def plot_sklearn_search(grid_search, param1, param2=None):
    """
    Plots a scatter plot showing the model score for the parameter combinations
    in a GridSearchCV object.

    Parameters:
    - grid_search: A fitted GridSearchCV object.
    - param1: The name of the first parameter.
    - param2: The name of the second parameter.
    """
    # Extract the results from the grid search
    results = grid_search.cv_results_
    param_data = results["param_" + param1].data

    # Check the type of 'param_data' and handle accordingly
    if isinstance(param_data, np.ndarray):
        param1_values = np.array(param_data, dtype=float)
        if param2 is not None:
            param2_values = np.array(results["param_" + param2].data, dtype=float)
    elif isinstance(param_data, list):
        param1_values = np.array(param_data, dtype=float)
        if param2 is not None:
            param2_values = np.array(param_data, dtype=float)

    elif isinstance(param_data, optuna.distributions.FloatDistribution):
        # Handle Optuna's FloatDistribution by extracting its lower/upper bounds
        param1_values = np.linspace(param_data.low, param_data.high, num=100)  # Example
        if param2 is not None:
            param2_values = np.linspace(param_data.low, param_data.high, num=100)

    else:
        raise ValueError(f"Unsupported type for 'param_data': {type(param_data)}")

    # Get parameter values for the two parameters
    param1_values = np.array(results["param_" + param1].data, dtype=float)

    # Extract the MSE scores (convert from negative MSE)
    mse_scores = -results["mean_test_score"]

    # Create the scatter plot
    fig = plt.figure(figsize=(10, 6))
    if param2 is None:
        scatter = plt.scatter(param1_values, mse_scores, cmap="viridis", s=100)
    else:
        scatter = plt.scatter(param1_values, param2_values, c=mse_scores, cmap="viridis", s=100)
        plt.ylabel(param2)

    plt.colorbar(scatter, label="Score")
    # Label the axes
    plt.xlabel(param1)

    plt.title("Score for different combinations of parameters")

    plt.show()
    plt.close(fig)


def plot_residuals(preds, valid_y, save_path=None):
    """
    Plots the residuals of the model predictions against the true values.

    Args:
    - model: The trained XGBoost model.
    - dvalid (xgb.DMatrix): The validation data in XGBoost DMatrix format.
    - valid_y (pd.Series): The true values for the validation set.
    - save_path (str, optional): Path to save the generated plot.
        If not specified, plot won't be saved.

    Returns:
    - None (Displays the residuals plot on a Jupyter window)
    """

    # Calculate residuals
    residuals = valid_y - preds

    # Set Seaborn style
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(valid_y, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    return fig


def plot_feature_importance(model, pipeline, x_train):
    """
    Plots feature importance for an XGBoost model or RandomForest.

    Args:
    - model: A trained XGBoost or RF model
    - pipeline: the pipeline that includes the hyperparameter optimization
    - x_train: the features

    Returns:
    - fig: The matplotlib figure object
    """

    step_names = [name for name, _ in pipeline.steps]
    if "selectfeatures" in step_names:
        features = x_train.columns
        included_check = pipeline["selectfeatures"].get_support()
        feature_names = [feature for feature, selected in zip(features, included_check) if selected]
        x_train_transformed = x_train[feature_names]
    else:
        x_train_transformed = x_train

    if isinstance(model, xgb.XGBRegressor):
        model.get_booster().feature_names = list(x_train_transformed.columns)
        fig, ax = plt.subplots(figsize=(10, 8))
        importance_type = (
            "weight"
            if pipeline["hyperparameter_optimization"].best_params_.get("booster") == "gblinear"
            else "gain"
        )
        xgb.plot_importance(
            model,
            importance_type=importance_type,
            ax=ax,
            title=f"Feature Importance based on {importance_type}",
        )
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Get feature importances
        importances = pipeline["hyperparameter_optimization"].best_estimator_.feature_importances_
        # Feature names
        important_features = pd.Series(data=importances, index=feature_names)
        important_features.sort_values(ascending=True, inplace=True)

        # Plot feature importances
        important_features.plot(kind="barh", figsize=(10, 8), ax=ax)
        plt.title("Feature Importance - RandomForestRegressor")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.xticks(rotation=45)
        plt.show()
        plt.close(fig)
    return fig


def plot_prediction_error(preds, y_valid):
    """
    Plots prediction error

    Args:
    - preds: predictions made by a certain model
    - y_valid: actual values of the dependent variable

    Returns:
    - A figure of the prediction errors
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.scatter(y_valid, preds)
    plt.plot(
        [min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color="red"
    )  # Perfect prediction line
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction Error Plot")
    plt.show()
    plt.close(fig)
    return fig


def plot_learning_curve(pipeline, x_train, y_train):
    """
    Plots learning curve

    Args:
    - preds: predictions made by a certain model
    - y_valid: actual values of the dependent variable

    Returns:
    - A figure of the learning curve
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    train_sizes, train_scores, test_scores = learning_curve(pipeline, x_train, y_train, cv=5)
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()
    plt.close(fig)
    return fig


def plot_validation_curve(pipeline, X, y, param_names, param_range):
    """
    Plots validation curve

    Args:
    - preds: predictions made by a certain model
    - y_valid: actual values of the dependent variable
    - param_names: names of the parameters for which you want to create a validation curve
    - param_range: parameter ranges

    Returns:
    - A figure of the validation curve
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    train_scores, test_scores = validation_curve(
        pipeline, X, y, param_name=param_names, param_range=param_range
    )
    plt.plot(param_range, -train_scores.mean(axis=1), label="Training Error")
    plt.plot(param_range, -test_scores.mean(axis=1), label="Validation Error")
    plt.xscale("log")
    plt.xlabel("Parameter Value")
    plt.ylabel("Error")
    plt.title("Validation Curve")
    plt.legend()
    plt.show()
    plt.close(fig)
    return fig


def plot_feature_importances_search(grid_search, X_train, y_train):
    """
    Plots the feature importance for each of the folds from a grid search

    Args:
    - grid_search: the grid search argument
    - x_train: the train data
    - y_train: the train data of the dependent variable

    Returns:
    prints a figure for each of the cross-validation folds of the feature importances
    """

    # Extract the results and models from GridSearchCV
    cv_results = grid_search.cv_results_
    best_index = grid_search.best_index_

    # Loop through each parameter combination and plot feature importance
    for i, (params, mean_score) in enumerate(
        zip(cv_results["params"], cv_results["mean_test_score"])
    ):
        # Get the model with the current parameter set
        model = (
            grid_search.best_estimator_
            if i == best_index
            else grid_search.estimator.set_params(**params).fit(X_train, y_train)
        )

        # Extract the feature importances
        feature_importances = model.feature_importances_

        # Sort feature importances for better visualization
        sorted_idx = np.argsort(feature_importances)
        sorted_features = np.array(X_train.columns)[sorted_idx]

        # Plotting the feature importance
        plt.figure(figsize=(8, 6))
        plt.barh(sorted_features, feature_importances[sorted_idx], color="skyblue")
        plt.xlabel("Feature Importance")
        plt.title(f"Feature Importance for Params: {params} (Mean Test Score: {mean_score:.4f})")

        # Show the plot
        plt.tight_layout()
        plt.show()


def plot_optuna_search(optuna_search, param1_name, param2_name=None):
    """
    Plots a scatter plot showing the Mean Squared Error (MSE) for the parameter combinations
    in an OptunaSearchCV object.

    Parameters:
    - optuna_search: A fitted OptunaSearchCV object.
    - param1_name: The name of the first parameter.
    - param2_name: The name of the second parameter (optional).
    """
    # Extract the study and trials from the OptunaSearchCV object
    study = optuna_search.study
    trials = study.trials

    # Get the parameter values and MSE scores for each trial
    param1_values = []
    param2_values = []
    mse_scores = []

    for trial in trials:
        if param1_name in trial.params:
            param1_values.append(trial.params[param1_name])
        if param2_name and param2_name in trial.params:
            param2_values.append(trial.params[param2_name])
        mse_scores.append(trial.value)  # Assumes the value is the negative MSE or a similar score

    param1_values = np.array(param1_values, dtype=float)
    mse_scores = np.array(mse_scores, dtype=float)

    # Create the scatter plot
    fig = plt.figure(figsize=(10, 6))

    if param2_name is None:
        # Plot only param1 vs MSE
        scatter = plt.scatter(param1_values, mse_scores, cmap="viridis", s=100)
    else:
        # Plot param1 and param2 vs MSE
        param2_values = np.array(param2_values, dtype=float)
        scatter = plt.scatter(param1_values, param2_values, c=mse_scores, cmap="viridis", s=100)
        plt.ylabel(param2_name)

    plt.colorbar(scatter, label="MSE")

    # Label the axes
    plt.xlabel(param1_name)
    plt.title("MSE for different combinations of parameters")

    # Show the plot
    plt.show()
    plt.close(fig)
