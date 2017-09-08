from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
import re
import sys
import os
import base64
import pickle


app = Flask(__name__)


#Loads the random forest model
#dest = os.path.join('randomforestflaskclassifier', 'pkl_model')
forest_clf = pickle.load(open(
                os.path.join('model',
                            'rforest_clf.pkl'), 'rb'))



#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	print ("debug")
	#read the image into memory
	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#print(x.shape , "This is the shape, bitch")
	x = x.flatten()
	result = forest_clf.predict([x])
	#print(result)
	response = np.array_str(result)
	return response
	#print(x)
	#print(x.shape)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

