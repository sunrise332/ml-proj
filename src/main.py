

import pandas as pd
from sklearn.pipeline import Pipeline

from config import config
from feature import add_cabin_known_col
from model import build_logistic_regression
from preprocessing import build_linear_preprocessor
from validation import run_cross_validation


def split_features_target(data, target_column):
    X_data = data.drop(columns=[target_column])
    y_tar_data = data[target_column]
    return X_data, y_tar_data

def run():
    train_data = pd.read_csv('./data/train/train.csv')
    test_data = pd.read_csv('./data/test/test.csv')
    passenger_ids = test_data['PassengerId']

    n_folds = config.validation.n_splits

    X_train, y_train = split_features_target(train_data, 'Survived')


    X_train = add_cabin_known_col(X_train)
    X_test = add_cabin_known_col(test_data)

    preprocessor = build_linear_preprocessor(
        numerical_features=list(config.features.numeric),
        categorical_features=list(config.features.categorical),
    )
    model = build_logistic_regression()

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    mean_accuracy, std_accuracy, _ = run_cross_validation(
        pipeline=pipeline,
        train_X=X_train,
        train_y=y_train,
        n_split=n_folds
    )
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Std accuracy: {std_accuracy:.4f}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    kaggle_submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': y_pred,
    })

    kaggle_submission.to_csv('./submission.csv', index=False)

    print(kaggle_submission.head())
    print(kaggle_submission.shape)


if __name__ == "__main__":
    run()

