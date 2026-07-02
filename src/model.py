import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def logistic_resgression(
    C: int = 0.1,
    l1_ratio: int = 0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
):
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=max_iter,
    )


def build_knn_classifier(n_neighbors: int=10, weights: str='uniform', metric: str='minkowski'):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)


def build_decision_tree_classifier(random_state: int = 42):
    return DecisionTreeClassifier(random_state=random_state)


def build_random_forest_classifier(random_state: int = 42):
    return RandomForestClassifier(random_state=random_state)


def build_xgboost_classifier(n_estimators: int = 150, colsample_bytree: float = 0.9, learning_rate: float = 0.09, max_depth: int = 5, min_child_weight: int = 2, subsample: float = 0.7):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        colsample_bytree=colsample_bytree,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
    )


def build_lightgbm_classifier(colsample_bytree: float = 0.9, learning_rate: float = 0.09, max_depth: int = 5, min_child_weight: int = 2, subsample: float = 0.7):
    return lgb.LGBMClassifier(
        colsample_bytree=colsample_bytree,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
    )


def build_catboost_classifier(
    depth: int = 3,
    iterations: int = 200,
    l2_leaf_reg: int = 7,
    learning_rate: float = 0.08,
    random_strength: float = 0.5,
    rsm: float = 0.9,
):
    return CatBoostClassifier(depth=depth, iterations=iterations, l2_leaf_reg=l2_leaf_reg, learning_rate=learning_rate, random_strength=random_strength, rsm=rsm)


def get_model(model_name: str, **kwargs):

    model_dict = {
        "logistic_resgression": logistic_resgression,
        "knn": build_knn_classifier,
        "decision_tree": build_decision_tree_classifier,
        "random_forest": build_random_forest_classifier,
        "xgboost": build_xgboost_classifier,
        "lightgbm": build_lightgbm_classifier,
        "catboost": build_catboost_classifier,
    }
    if model_name in model_dict:
        model_builder = model_dict[model_name]
        return model_builder(**kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(model_dict.keys())}"
        )