import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return {"accuracy": acc}
