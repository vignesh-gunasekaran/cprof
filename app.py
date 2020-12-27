from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
import pandas as pd
import xgboost as xg

app = Flask(__name__)

model1 = pickle.load(open('cancer_model.pkl', 'rb'))


@app.route('/')
@app.route('/Home')
def home():
   return render_template('home.shtml')


@app.route('/Cancer Prediction')
def cancer():
    return render_template('cancer.shtml')


@app.route('/Liver Diagnosis')
def liver():
    return render_template('liver.shtml')


@app.route('/Heart Issue')
def heart():
    return render_template('heart.shtml')


@app.route('/About')
def about():
    return render_template('aboutme.shtml')


@app.route('/predict1', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model1.predict(final)

    if prediction == 0:
        return render_template('cancer.shtml',pred='The cancer is benign')
    else:
        return render_template('cancer.shtml',pred='The cancer is malignant')

#@app.route('/predict2', methods=['POST', 'GET'])
#def predict2():
#    int_features2 = [float(x) for x in request.form.values()]
#    final2 = [np.array(int_features2)]
#    print(int_features2)
#    print(final2)
#    prediction2 = model2.predict(final2)

 #   if prediction2 == 0:
  #      return render_template('heart.shtml', pred='There is no damage in Heart')
  #  else:
  #      return render_template('heart.shtml', pred='There is serious damage in heart, which might result in heart failure')



