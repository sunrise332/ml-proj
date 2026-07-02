from omegaconf import OmegaConf

config = {
    "general": {
        "random_state": 42,
        "active_model": "logistic_resgression",
        "train_data_path": "./data/train/train.csv",
        "test_data_path": "./data/test/test.csv",
    },
    "validation": {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },

    "features": {
        "numeric": [
            "Age",
            "Fare",
            "SibSp",
            "Parch",
        ],
        "categorical": [
            "Sex",
            "Embarked",
            "Pclass",
            "CabinKnown",
        ],
        "drop": [
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin",
        ],
    },

    "preprocessing": {
        "numeric_imputer_strategy": "median",
        "categorical_imputer_strategy": "most_frequent",
        "scale_numeric": True,
        "one_hot_drop": "first",
    },

    "model": {
        "logistic_resgression": {
            "C": 1.0,
            "l1_ratio": 0.0,
            "solver": "lbfgs",
            "max_iter": 1000,
        },

        "knn": {
            "n_neighbors": 14,
            "weights": "uniform",
            "metric": "manhattan",
        },
        
        "decision_tree": {
        "criterion": "entropy",
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "random_state": 42,
        "param_grid": {
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 3, 4],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
        },
        },
        "xgboost": {
            "n_estimators": 150,
            "learning_rate": 0.09,
            "max_depth": 5,
            "min_child_weight": 2,
            "subsample": 0.7,
            "colsample_bytree": 0.9,
        },
        "lightgbm": {
            "n_estimators": 150,
            "learning_rate": 0.09,
            "max_depth": 5,
            "min_child_weight": 2,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbosity": -1,
        },
        "catboost": {
            "iterations": 200,
            "learning_rate": 0.08,
            "depth": 3,
            "l2_leaf_reg": 7,
            "random_strength": 0.5,
            "rsm": 0.9,
            "random_seed": 42,
            "verbose": 0,
        },
    },
}

config = OmegaConf.create(config)