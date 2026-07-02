from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_scaled_preprocessor(numerical_features, categorical_features):
    imputer_num = SimpleImputer(strategy="median")
    imputer_cat = SimpleImputer(strategy="most_frequent")
    one_hot = OneHotEncoder(drop="first", handle_unknown="ignore")
    scaler = StandardScaler()

    numeric_pipeline = Pipeline(steps=[("imputer", imputer_num), ("scaler", scaler)])
    categorical_pipeline = Pipeline(steps=[("imputer", imputer_cat), ("encoder", one_hot)])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_pipeline, numerical_features), ("cat", categorical_pipeline, categorical_features)], remainder="drop")

    return preprocessor


def build_tree_preprocessor(numerical_features, categorical_features):
    imputer_num = SimpleImputer(strategy="median")
    imputer_cat = SimpleImputer(strategy="most_frequent")
    one_hot = OneHotEncoder(drop="first", handle_unknown="ignore")

    numeric_pipeline = Pipeline(steps=[("imputer", imputer_num)])
    categorical_pipeline = Pipeline(steps=[("imputer", imputer_cat), ("encoder", one_hot)])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_pipeline, numerical_features), ("cat", categorical_pipeline, categorical_features)], remainder="drop")

    return preprocessor


def get_preprocessor(model_name: str, numerical_features, categorical_features):
    '''
    Получаем препроцессор по названию модели. Если названия модели нет в множестве, падаем с ошибкой
    '''
    scaled_models = {
        "logistic_resgression",
        "knn",
    }
    tree_models = {
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    }
    if model_name in scaled_models:
        return build_scaled_preprocessor(numerical_features, categorical_features)
    elif model_name in tree_models:
        return build_tree_preprocessor(numerical_features, categorical_features)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"There is no a such preprocessor for that model"
        )



