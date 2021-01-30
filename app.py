
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:50:10 2020

@author: kosaraju vivek
"""

import numpy as np
import pandas as pd
from flask import Flask, request,  render_template
import pickle


app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        in1 = float(request.form['satisfaction_level'])
        in2 = float(request.form['last_evaluation'])
        in3 = int(request.form['number_project'])
        in4 = int(request.form['average_montly_hours'])
        in5 = int(request.form['time_spend_company'])
        in6 = int(request.form['Work_accident'])
        in7 = int(request.form['promotion_last_5years'])
        in8 = str(request.form['Departments'])
        in9 = str(request.form['salary'])
        x = [[in1,in2,in3,in4,in5,in6,in7,in8,in9],[in1,in2,in3,in4,in5,in6,in7,in8,in9],]
        x = np.array(x)
        x[:,8] = encoder.transform(x[:,8])
        x[:,0:5] = scaler.transform(x[:,0:5])
        x = transformer.transform(x)
        pred = model.predict(x)
        if pred[0] == 0:
            output = "not leave"
            return render_template('index.html', your_caption =output)
        else:
            output = "leave"
            return render_template('index.html', your_caption =output)
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)