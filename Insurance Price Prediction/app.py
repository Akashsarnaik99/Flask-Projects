from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('models/insuarance_model.pkl','rb'))
smoker_encoder = pickle.load(open('models/smoker_encoder.pkl','rb'))
poly_features = pickle.load(open('models/poly_features.pkl','rb'))





app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict_price',methods=['GET','POST'])
def predict_price():
    age = int(request.form.get('Age'))
    children = int(request.form.get('Children'))
    bmi = (request.form.get('bmi'))
        
    smoker = request.form.get('smoker')
    msg=None
    try:
        bmi=float(bmi)
    except ValueError:
        msg = "Please enter correct value of BMI"
    if msg:
        return render_template('index.html', msg=msg)


    if smoker =='yes':
        smoker = 1
    elif smoker=='no':
        smoker=0
    
    result=None
    input = [[age,bmi,children,smoker]]
    poly_input =poly_features.transform(input)
    result = model.predict(poly_input)
    result=np.round(result[0],2)


    return render_template('index.html',result=result,msg=msg )







if __name__ == '__main__':
    app.run(debug=True)
