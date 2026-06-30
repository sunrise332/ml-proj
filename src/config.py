from omegaconf import OmegaConf

config = {
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
    },
}

config = OmegaConf.create(config)