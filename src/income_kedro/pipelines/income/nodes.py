import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    return df


def split_data(df: pd.DataFrame, split_params: dict):
    X = df.drop("income", axis=1)
    y = df[["income"]]

    return train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )

def fit_preprocessor(X_train: pd.DataFrame, feature_params) -> ColumnTransformer:
    X_train = X_train.copy()

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
    y = y.copy()

    target_col = feature_params['target_col']
    y[target_col] = y[target_col].apply(lambda x: 1 if x == '>50K' else 0)

    return y

def apply_preprocessor(X: pd.DataFrame, y: pd.DataFrame, preprocessor: ColumnTransformer, feature_params) -> pd.DataFrame:
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
    model = LogisticRegression(
        C=model_params["C"],
        penalty=model_params["penalty"],
        solver=model_params["solver"],
        random_state=model_params["random_state"],
    )
    model.fit(X_train, y_train)
    return model


def _gen_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    
    return {
        "f1_score": f1,
        "precision": pr,
        "recall": rc
        }


def _store_pr_curve(model, X_test, y_test):
    y_score = model.predict_proba(X_test)[:, 1]
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


def _store_confusion_matrix(model ,X_test, y_test):
    y_pred = model.predict(X_test)
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


def evaluate_model(model, X_test, y_test):
    metrics = _gen_metrics(model, X_test, y_test)
    _store_pr_curve(model, X_test, y_test)
    _store_confusion_matrix(model, X_test, y_test)

    return metrics
