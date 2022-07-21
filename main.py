from flask import Flask,jsonify,request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from PIL import Image

app=Flask(__name__)

X = np.load("D:\programing\data\\aLpHaBet ReCogNiTioN\image.npz")["arr_0"]
Y = pd.read_csv("D:\programing\data\\aLpHaBet ReCogNiTioN\\a.csv")["labels"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8,random_state=0)

X_train = X_train/255.0
X_test = X_test/255.0

classifier = LogisticRegression(solver="saga",multi_class="multinomial")
classifier.fit(X_train,Y_train)

def get_pred(image):
    im_pil = Image.open(image)
    image_ = im_pil.convert('L')
    image_resized = image_.resize((22,30), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_resized,pixel_filter)
    image_resized_invert = np.clip(image_resized-min_pixel,0,255)
    max_pixel = np.max(image_resized)
    image_resized_invert_scaled = np.asarray(image_resized_invert_scaled)/max_pixel
    sample = np.array(image_resized_invert_scaled).reshape(1,660)
    pred = classifier.predict(sample)
    return pred

@app.route("/")
def home():
    return"""
    <body style=\"background-image:linear-gradient(to top right,blue,pink)\">
        <h1 style="color:white">alphabet prediction</h1>
    </body>
    """

@app.route("/predict",methods=["POST"])
def predict():
    if not request.json:
        return jsonify({
            "status":"error",
            "message":"please provide a data!"
        },400)

    prediction = predict(request.files.get("alphabet"))
    return f"""
    <body style="background-image:linear-gradient(to right,#06beb6,#48b1bf)">
    <h1>prediction{prediction}</h1>
    </body>
    """
    
    



if (__name__ == "__main__"):
    app.run(debug=True)