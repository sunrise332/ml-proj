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
    train_fold_accuracy = []
    val_fold_accuracy = []

    for train_indices, val_indices in skf.split(train_X, train_y):
        x_train_fold = train_X.iloc[train_indices]
        x_val_fold = train_X.iloc[val_indices]

        y_train_fold = train_y.iloc[train_indices]
        y_val_fold = train_y.iloc[val_indices]

        pipeline.fit(x_train_fold, y_train_fold)

        y_train_pred = pipeline.predict(x_train_fold)
        y_val_pred = pipeline.predict(x_val_fold)
        
        train_accuracy = accuracy_score(y_train_fold, y_train_pred)
        val_accuracy = accuracy_score(y_val_fold, y_val_pred)

        train_fold_accuracy.append(train_accuracy)
        val_fold_accuracy.append(val_accuracy)

    mean_val_accuracy = sum(val_fold_accuracy) / len(val_fold_accuracy)
    std_val_accuracy = np.std(val_fold_accuracy)
    mean_train_accuracy = sum(train_fold_accuracy) / len(train_fold_accuracy)
    return mean_val_accuracy, std_val_accuracy, mean_train_accuracy, val_fold_accuracy, train_fold_accuracy