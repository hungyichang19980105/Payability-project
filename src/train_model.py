import hydra
import numpy as np
import pandas as pd
import lightgbm as lgbm
import pathlib
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


@hydra.main(config_path="config", config_name="main")
def train_model(config: DictConfig):

    # get current path
    current_path = hydra.utils.get_original_cwd() + "/"
    ratio = config.processed.ratio

    # Read data
    df = pd.read_csv(current_path + config.processed.path,
                     dtype=config.processed.type)
    df = df.drop(df.columns[[0]], axis=1)

    # train_test_split
    training_set, test_set = np.split(df, [int(ratio * len(df))])

    X_train = training_set.drop("is_fraud", axis=1)
    y_train = training_set[['is_fraud']].values.flatten()

    X_test = test_set.drop("is_fraud", axis=1)
    y_test = test_set[['is_fraud']].values.flatten()

    numeric_features = list(config.variables.num_idx)
    categorical_features = list(config.variables.cat_idx)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')
         ), ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features), ('categorical',
                                                                 categorical_transformer, categorical_features)
        ], remainder="drop")

    if config.model.name == 'RandomForest Classifier':
        model = RandomForestClassifier(
            n_estimators=config.model.n_estimators, max_depth=config.model.max_depth)
    elif config.model.name == 'LightGBM Classifier':
        model = lgbm.LGBMClassifier(objective=config.model.objective,
                                    n_estimators=config.model.n_estimators, learning_rate=config.model.learning_rate)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), ('classifier', model)
    ])

    model = pipeline.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Training on model " + config.model.name + "......")
    print(classification_report(predictions, y_test))


if __name__ == "__main__":
    train_model()
