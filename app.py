#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 18:32:59 2022

@author: Soham Rajput
"""

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
pipeline = joblib.load('models/pipeline.sav')
model = ''
model_name = ''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        input_dict = request.form.to_dict()
        model_selector = input_dict.pop('Model')
        features_df = pd.DataFrame.from_dict([input_dict])
        transformed_input = pipeline.transform(features_df)

        if model_selector == 'Logistic':
            model_name = 'Logistic Regression'
            model = joblib.load('models/logistic_model.sav')
        elif model_selector == 'SVM':
            model_name = 'Support Vector Machine'
            model = joblib.load('models/svm_model.sav')
        elif model_selector == 'DT':
            model_name = 'Decision Tree'
            model = joblib.load('models/decision_tree_model.sav')
        elif model_selector == 'NN':
            model_name = 'Multilayer Perceptron'
            model = joblib.load('models/nn_model.sav')

        predict_output = model.predict(transformed_input)
        print('Model used:', model_name)
        print('Prediction:', predict_output)

    return render_template('result.html', model_name=model_name, prediction=predict_output)

if __name__ == "__main__":
    app.run(debug=True)