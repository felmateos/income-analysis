import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing values from the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with rows containing NaN values removed.
    """

    df = df.dropna()
    return df


def split_data(df: pd.DataFrame, split_params: dict):
    """
    Split the dataset into training and testing sets.

    The target variable is assumed to be the column named 'income'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and target column.
    split_params : dict
        Dictionary containing:
            - test_size (float): Proportion of the dataset to include in the test split.
            - random_state (int): Random seed for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    X = df.drop("income", axis=1)
    y = df[["income"]]

    return train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )

def fit_preprocessor(X_train: pd.DataFrame, feature_params) -> ColumnTransformer:
    """
    Fit a preprocessing pipeline for categorical and numerical features.

    Categorical features are one-hot encoded (dropping first level),
    and numerical features are standardized.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    feature_params : dict
        Dictionary containing:
            - cat_cols (list): List of categorical column names.
            - num_cols (list): List of numerical column names.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessing transformer.
    """

    X_train = X_train.copy()

    print(X_train.columns)

    cat_cols = feature_params['cat_cols']
    num_cols = feature_params['num_cols']
    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols),
        ],
        remainder='drop'
    )

    ct.fit(X_train)

    return ct

def _encode_label(y: pd.DataFrame, feature_params) -> pd.DataFrame:
    """
    Encode the target variable into binary format.

    The target is converted to:
        '>50K' -> 1
        otherwise -> 0

    Parameters
    ----------
    y : pd.DataFrame
        Target DataFrame.
    feature_params : dict
        Dictionary containing:
            - target_col (str): Name of the target column.

    Returns
    -------
    pd.DataFrame
        Encoded target DataFrame.
    """

    y = y.copy()

    target_col = feature_params['target_col']
    y[target_col] = y[target_col].apply(lambda x: 1 if x == '>50K' else 0)

    return y

def apply_preprocessor(X: pd.DataFrame, y: pd.DataFrame, preprocessor: ColumnTransformer, feature_params) -> pd.DataFrame:
    """
    Apply a fitted preprocessor to features and encode the target.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Target DataFrame.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    feature_params : dict
        Dictionary containing:
            - target_col (str): Name of the target column.

    Returns
    -------
    tuple
        Transformed feature DataFrame and encoded target DataFrame.
    """

    X_t = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X = pd.DataFrame(
        X_t,
        columns=feature_names,
        index=X.index
    )
    y = _encode_label(y, feature_params)

    return X, y


def train_model(X_train, y_train, model_params):
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target values.
    model_params : dict
        Dictionary containing:
            - C (float): Inverse regularization strength.
            - penalty (str): Type of regularization.
            - solver (str): Optimization solver.
            - random_state (int): Random seed.

    Returns
    -------
    LogisticRegression
        Fitted Logistic Regression model.
    """

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def _predict(model, model_configs, X_test):
    """
    Runs inference for a classification model using a specified threshold.

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    model_configs : dict
        Dictionary containing model configuration parameters (mainly threshold).
    X_test : array-like
        Test feature matrix.

    Returns
    -------
    dict
        Predicted class labels based on the specified threshold.
    """
    threshold = model_configs['threshold']
    scores = model.predict_proba(X_test)[:, 1]
    return (scores >= threshold).astype(int)


def _predict_proba(model, X_test):
    """
    Runs probability inference for a classification model.

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    X_test : array-like
        Test feature matrix.

    Returns
    -------
    dict
        Probability of predicted class labels based on the specified threshold.
    """
    return model.predict_proba(X_test)


def _gen_metrics(model, model_configs, X_test, y_test):
    """
    Compute classification performance metrics.

    Metrics computed:
        - F1-score
        - Precision
        - Recall

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    dict
        Dictionary containing f1_score, precision, and recall.
    """

    y_pred = _predict(model, model_configs, X_test)

    f1 = f1_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    
    return {
        "f1_score": f1,
        "precision": pr,
        "recall": rc
        }


def _store_pr_curve(model, model_configs, X_test, y_test): # pragma: no cover
    """
    Generate and save the Precision-Recall curve.

    The curve is saved to 'images/pr_curve.png'.

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model with predict_proba method.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    None
    """

    y_score = _predict_proba(model, X_test)[:, 1]
    pr, rc, _ = precision_recall_curve(y_test, y_score)
    auc_ = auc(rc, pr)

    plt.figure(figsize=(7, 7))

    plt.plot(rc, pr, label=f'(AUC = {auc_:.3f})', c='green')

    baseline = y_test.mean()
    plt.hlines(
        y=baseline,
        xmin=0,
        xmax=1,
        linestyles='--',
        label=f'Aleatório (prevalência = {baseline[0]:.2f})'
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('equal')

    ticks = np.linspace(0, 1, 6)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend()
    
    plt.savefig(
        "images/pr_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    return


def _store_confusion_matrix(model, model_configs, X_test, y_test): # pragma: no cover
    """
    Generate and save the normalized confusion matrix heatmap.

    The matrix is normalized by true labels and saved to
    'images/confusion_matrix.png'.

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    None
    """

    y_pred = _predict(model, model_configs, X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    plt.figure(figsize=(9, 7))

    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens')

    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.savefig(
        "images/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )


def evaluate_model(model, model_configs, X_test, y_test):
    """
    Evaluate a trained model and store evaluation artifacts.

    This function:
        - Computes classification metrics
        - Saves Precision-Recall curve
        - Saves confusion matrix

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    metrics = _gen_metrics(model, model_configs, X_test, y_test)
    _store_pr_curve(model, model_configs, X_test, y_test)
    _store_confusion_matrix(model, model_configs, X_test, y_test)

    return metrics
