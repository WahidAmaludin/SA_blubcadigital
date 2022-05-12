from flask import Flask, request, render_template, redirect
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

#load model
mnb_s = pickle.load(open("weight_pred_model_pkl.pkl", 'rb'))


cv = pickle.load(open("cv_pkl.pkl", 'rb'))
#flask

@app.route("/")
def index():
    return render_template('main.html')
	
@app.route("/result", methods = ["GET","POST"])
def result():
	if request.method == 'POST':
		text = request.form['teks']
		data = [text]
		tf = cv.transform(data).reshape(1, -1)
		hasil = mnb_s.predict(tf)
		return render_template("result.html",result=hasil)

if __name__ == "__main__":
    app.run()