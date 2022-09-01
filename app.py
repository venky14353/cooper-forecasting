# coding: utf-8

import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
#from sklearn import metrics
from flask import Flask, request, render_template
#import re
#import math
import pickle
#from xgboost import XGBRegressor

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")



@app.route("/predict", methods=['POST'])
def predict():
    
    model = pickle.load(open("RF_tuned.sav", "rb"))
    

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']

    
    
    
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, inputQuery8]]
    
    # Create the pandas DataFrame 
    new_df = pd.DataFrame(data, columns = ['Week No', 'Year', 'Mortgage Rate', 'Treasury Yield','Unemployment Rate', 'GDP', 'Consumer Confidence Index','Disposable Income'])
    prediction = model.predict(new_df)
    o1 = "The forecasting value is {}".format(prediction)
    
    return render_template('home.html', output1=o1, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],
                           query5 = request.form['query5'],query6 = request.form['query6'],query7 = request.form['query7'],query8 = request.form['query8'])
    
if __name__ == "__main__":
    app.run(debug=True)
