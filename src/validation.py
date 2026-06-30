import numpy as np
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def run_cross_validation(pipeline: Pipeline,
                       train_X: DataFrame, 
                       train_y: Series, 
                       n_split: int = 5, 
                       random_state: int = 42,
                       shuffle: bool = True):
    '''Запускает кросс-валидацию используя подход stratified k fold'''
    
    skf = StratifiedKFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    fold_accuracy = []

    for train_indices, val_indices in skf.split(train_X, train_y):
        x_train_fold = train_X.iloc[train_indices]
        x_val_fold = train_X.iloc[val_indices]

        y_train_fold = train_y.iloc[train_indices]
        y_val_fold = train_y.iloc[val_indices]

        pipeline.fit(x_train_fold, y_train_fold)

        y_pred = pipeline.predict(x_val_fold)

        accuracy = accuracy_score(y_val_fold, y_pred)
        fold_accuracy.append(accuracy)

    mean_accuracy = sum(fold_accuracy) / len(fold_accuracy)
    std_accuracy = np.std(fold_accuracy)
    return mean_accuracy, std_accuracy, fold_accuracy