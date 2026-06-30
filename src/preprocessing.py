from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_linear_preprocessor(
        numerical_features,
        categorical_features
):
    imputer_num = SimpleImputer(strategy='median')   
    imputer_cat = SimpleImputer(strategy='most_frequent')
    one_hot = OneHotEncoder(drop='first', handle_unknown='ignore')
    scaler = StandardScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', imputer_num),
            ('scaler', scaler)
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', imputer_cat),
            ('encoder', one_hot)
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor
