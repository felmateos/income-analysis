from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_data,
    split_data,
    fit_preprocessor,
    apply_preprocessor,
    train_model,
    evaluate_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(clean_data, "raw_income", "clean_income"),
            node(
                split_data,
                inputs=dict(
                    df="clean_income",
                    split_params="params:split_params",
                ),
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),
            node(
                fit_preprocessor,
                inputs=["X_train", "params:feature_params"],
                outputs="preprocessor"
            ),
            node(
                apply_preprocessor,
                inputs=["X_train", "y_train", "preprocessor", "params:feature_params"], 
                outputs=["X_train_features", "y_train_features"]),
            node(
                apply_preprocessor,
                inputs=["X_test", "y_test", "preprocessor", "params:feature_params"], 
                outputs=["X_test_features", "y_test_features"]),
            node(
                train_model,
                inputs=["X_train_features", "y_train_features", "params:model_params"],
                outputs="model",
            ),
            node(
                evaluate_model,
                inputs=["model", "X_test_features", "y_test_features"],
                outputs="metrics",
            ),
        ]
    )
