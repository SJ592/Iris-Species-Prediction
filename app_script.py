import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    filepath=r'D:/obj/iris_model.pkl'
    loaded_model = pickle.load(open(filepath,"rb"))
    result = loaded_model.predict(to_predict)
    return result

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()       #entered form values are stored here in form of dictionary
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)==0:
            prediction='Iris Setosa'
        elif int(result)==1:
            prediction='Iris Versicolor'
        else:
            prediction='Iris Virginica'
            
        return flask.render_template("result.html",prediction=prediction)
    
if __name__=='__main__':
	app.run()