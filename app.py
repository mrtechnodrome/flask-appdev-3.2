from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('linear_regression_model_sc.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    GRE_Score = int(request.form.get('GRE_Score'))
    TOEFL_Score = int(request.form.get('TOEFL_Score'))
    University_Rating = int(request.form.get('University_Rating'))
    CGPA = float(request.form.get('CGPA'))
    Research = int(request.form.get('Research'))

    input_query = np.array([[GRE_Score,TOEFL_Score,University_Rating,CGPA,Research]])

    Chance_of_Admit = model.predict(input_query)[0]

    return jsonify({'Chance_of_Admit':float(Chance_of_Admit)})

if __name__ == '__main__':
    app.run(debug=True)