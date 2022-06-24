import pandas as pd
import numpy as np
from flask import Flask, url_for, redirect, request, jsonify, render_template
import pickle
from pycaret.regression import *

app = Flask(__name__) #Initialize the flask App
model = load_model('modelv1')
cols = ['host_total_listings_count','new_property_type','room_type','accommodates','bathrooms_count','Shared/Category','bedrooms','beds','minimum_nights','maximum_nights','availability_30','neigbourhood_level','amenities_num','amen_group_kitchen','amen_group_cleaning','amen_group_safety','amen_group_household','amen_group_bedroom','amen_group_electronics','amen_group_extra_spaces']

@app.route('/')
def home():
    return render_template('/main.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen)
    prediction = int(prediction.Label[0])
    prediction = np.exp(prediction) 
    prediction = np.round(prediction, 2) 
    
    return render_template('main.html', prediction_text='Rent should be +/- $10 of $ {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)