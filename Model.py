import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import pandas as pd
import os
datapath = r"C:\Users\venkateshwarar\Desktop\firstsource\cooper"
os.chdir(datapath)
import seaborn as sns
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBRegressor

d1=pd.read_excel('Weekly data Refinance Volumes_upto_aug1.xlsx')


future_data  = d1[d1['Refinance'].isna()]



base_data = d1[~d1['Refinance'].isna()]


base_data=base_data.drop('Date',axis=1)

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation_check(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


corr_features = correlation_check(x, 0.8)


base_data_upd = base_data.drop(corr_features,axis=1)


x=base_data_upd.drop('Refinance',axis=1)
y=base_data_upd['Refinance']


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=24)


def hyperParameterTuning_xgboost(X_train, y_train):
    
    param_tuning = {
        'learning_rate': [0.01,0.05,0.1,0.15,0.20,0.25,0.30],
        'max_depth': [3, 5, 7, 10,15,20,25,30],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.3,0.5,0.7,0.8],
        'colsample_bytree': [0.3,0.5,0.7,0.8],
        'n_estimators' : [int(x) for x in np.linspace(start = 10, stop = 180, num = 10)],
        'objective': ['reg:squarederror']
    }
    print(param_tuning)
    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

#hyperParameterTuning_xgboost(X_train, y_train)


xgb_model = XGBRegressor(
        objective = 'reg:squarederror',
        colsample_bytree = 0.7,
        learning_rate = 0.15,
        max_depth = 7,
        min_child_weight = 1,
        n_estimators = 28,
        subsample = 0.8)


xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test,y_test)], verbose=False)



def model_validation_xgboost(x,X_train,X_test,y_train,y_test):
    validation_table=[]
    models=[XGBRegressor]
    for i in models:
        rd=i(objective = 'reg:squarederror',
            colsample_bytree = 0.7,
            learning_rate = 0.1,
            max_depth = 10,
            min_child_weight = 1,
            n_estimators = 30,
            subsample = 0.7)               
        rd.fit(X_train,y_train)
        if x=="train":
            rd.predict(X_train)
            mae=mean_absolute_error(y_train.values,rd.predict(X_train))
            mse=mean_squared_error(y_train.values,rd.predict(X_train))
            rmse=np.sqrt(mean_squared_error(y_train.values,rd.predict(X_train)))
            mape=(((abs(y_train.values-rd.predict(X_train))/y_train.values)*100).sum())*(1/y_train.shape[0])
            rsquare=r2_score(y_train.values,rd.predict(X_train))
            validation_table.append([i,mae,mse,rmse,mape,rsquare])
        #validation_table=pd.DataFrame(validation_table_train)
        #validation_table.columns=["model name","mae","mse","rmse","mape","rsquare"]
        else:
            rd.predict(X_test)
            mae=mean_absolute_error(y_test.values,rd.predict(X_test))
            mse=mean_squared_error(y_test.values,rd.predict(X_test))
            rmse=np.sqrt(mean_squared_error(y_test.values,rd.predict(X_test)))
            mape=(((abs(y_test.values-rd.predict(X_test))/y_test.values)*100).sum())*(1/y_test.shape[0])
            rsquare=r2_score(y_test.values,rd.predict(X_test))
            validation_table.append([i,mae,mse,rmse,mape,rsquare])
            
    validation_table=pd.DataFrame(validation_table)
    validation_table.columns=["model name","mae","mse","rmse","mape","rsquare"]
    return validation_table

val_table_xgboost_train = model_validation_xgboost('train',X_train,X_test,y_train,y_test)
val_table_xgboost_train

val_table_xgboost_test = model_validation_xgboost('test',X_train,X_test,y_train,y_test)
val_table_xgboost_test


# save the model to disk
filename = 'xgboost_tuned.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

def predictions_on_future_data(future_data):
    pred_data = future_data[['Week No', 'Year', 'Mortgage Rate', 'Treasury Yield',
       'Unemployment Rate', 'GDP', 'Consumer Confidence Index',
       'Disposable Income']]
    filename = 'xgboost_tuned.sav'
    rf_model = pickle.load(open(filename, 'rb'))
    y_pred = rf_model.predict(pred_data)
    print(y_pred)
    future_data['Purchase_Pred'] = y_pred
    
    return future_data


pred_data = future_data[['Week No', 'Year', 'Mortgage Rate', 'Treasury Yield',
       'Unemployment Rate', 'GDP', 'Consumer Confidence Index',
       'Disposable Income']]


preds_tuned = predictions_on_future_data(future_data)

preds_tuned.to_csv("preds_on_future_data.csv",index=False)



