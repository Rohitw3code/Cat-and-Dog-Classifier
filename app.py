import pandas as pd
import numpy as np
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import InputLayer,Dense,Flatten,Conv2D,Activation,MaxPooling2D,Dropout
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import load_img
from flask import Flask,render_template,url_for,request,app,jsonify

app = Flask(__name__)

import tensorflow as tf
new_model = tf.keras.models.load_model('saved_model/my_model')
# Check its architecture
#new_model.summary()
@app.route("/")
def home():
	return render_template("home.html")

@app.route("/predict_image",methods=["POST"])
def predict_image():
    file = request.files['file']
    filename = file.filename
    if str(filename).strip():
		image = load_img('dog.jpg', target_size=(256, 256))
		img = np.array(image)
		img = img / 255.0
		img = img.reshape(1,256,256,3)
		label = model.predict(img)
		print("Predicted Class (0 - Cat , 1- Dog): ",round(label[0][0]))
		val = round(label[0][0])
		pred = "Cat" if val else "Dog"
        return render_template("home.html",prediction=val,img_name=filename)
    else:
        return render_template("home.html")


if __name__ == "__main__":
	app.run(debug=True)
