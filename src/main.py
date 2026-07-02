from omegaconf import OmegaConf
import pandas as pd
from sklearn.pipeline import Pipeline

from config import config
from feature import add_cabin_known_col
from model import get_model
from preprocessing import get_preprocessor
from validation import run_cross_validation


def split_features_target(data, target_column):
    X_data = data.drop(columns=[target_column])
    y_tar_data = data[target_column]
    return X_data, y_tar_data


def run():
    train_data = pd.read_csv(config.general.train_data_path)
    test_data = pd.read_csv(config.general.test_data_path)

    n_folds = config.validation.n_splits

    X_train, y_train = split_features_target(train_data, "Survived")

    X_train = add_cabin_known_col(X_train)
    X_test = add_cabin_known_col(test_data)
    
    # model creation
    model_params = OmegaConf.to_container(
        config.model['logistic_resgression'],
        resolve=True,
    )
    model_name = config.general.active_model
    model = get_model(model_name=config.general.active_model, **model_params)
    print(model)

    # prerocessing
    preprocessor = get_preprocessor(
        model_name=model_name,
        numerical_features=list(config.features.numeric),
        categorical_features=list(config.features.categorical)
        )
    
    # pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # check metrics. should write in file
    mean_val_accuracy, std_val_accuracy, mean_train_accuracy, _, _ = run_cross_validation(pipeline=pipeline, train_X=X_train, train_y=y_train, n_split=n_folds)
    print(f"Mean val accuracy: {mean_val_accuracy:.4f}")
    print(f"Std val accuracy: {std_val_accuracy:.4f}")
    print(f"Mean train accuracy: {mean_train_accuracy:.4f}")

    # TODO: for trees logic (вынести в другие файлы)
    # param_grid = OmegaConf.to_container(
    #     config.model.decision_tree.param_grid,
    #     resolve=True,
    # )
    # param_grid = {f"model__{name}": values for name, values in param_grid.items()}
    # grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=1, return_train_score=True, verbose=2)

    # learning
    # pipeline.fit(X_train, y_train)
    # grid_search.fit(X_train, y_train)

    # prediction
    y_pred = pipeline.predict(X_test)
    # y_pred = grid_search.best_estimator_.predict(X_test)
    # print(f"Best cross val score: {grid_search.best_score_:4f}")

    # save data
    kaggle_submission = pd.DataFrame(
        {
            "PassengerId": test_data["PassengerId"],
            "Survived": y_pred,
        }
    )
    kaggle_submission.to_csv("./submission.csv", index=False)


if __name__ == "__main__":
    run()
