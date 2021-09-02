# import
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import argparse
import config, models

def run(df, file, model):

    # split df into features (X) and target(y)
    X = df.drop(columns=['id','target']).copy()
    y = df.target

    # OneHOt encoding for categorical variables
    OH_encoder = OneHotEncoder(handle_unknown='return_nan', return_df=True, use_cat_names=True)

    # initiate model
    clf = models.models[model]

    # pipeline
    my_pipeline = Pipeline(steps=[('OHE', OH_encoder), ('model', clf)])

    # train-test split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0

    for train_index, test_index in kf.split(X, y):

        print ('fold:', fold)
        
        X_train = X.iloc[train_index]
        X_valid = X.iloc[test_index]
        
        y_train = y.iloc[train_index]
        y_valid = y.iloc[test_index]

        # fit model on training data

        print ('fitting train data')
        my_pipeline.fit(X_train, y_train)

        # predict on validation data

        print('predicitng test data')
        pred = my_pipeline.predict(X_valid)

        # score

        print ('calculating score')
        mse = mean_squared_error(y_valid, pred)
        rmse = np.sqrt(mse)

        # write score to file
        file.write(f'Fold: {fold}, RMSE: {rmse} \n')

        fold +=1
        print ('complete')
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--filename', type=str)

    args = parser.parse_args()

    train = pd.read_csv(config.TRAIN_CSV)

    file = open(os.path.join(config.OUTPUT, args.filename), 'w')

    run(train, file, model = args.model)

    file.close()



