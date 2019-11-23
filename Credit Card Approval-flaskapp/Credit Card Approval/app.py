from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
	#Alternative Usage of Saved Model
    RF_Model = open('Random_Forrest_Credit_Approval.pkl','rb')
    rf = joblib.load(RF_Model)

    if request.method == 'POST':
        # Fetch value for Gender
        gender = request.form['Gender']
        if gender == "Male":
            gender = 1.0
        else:
            gender = 0.0

        # Featch Value for Age
        age = float(request.form['Age'])

        # Fetch value for  Total Debt
        debt = float(request.form['Total_Debt'])

        # Fetch value for Total Professional Experience
        experience = float(request.form['Tot_Experience'])

        # Fetch value for Total Professional Experience
        income = float(request.form['Income'])

        # Fetch value for Bank Customer
        b_cust = request.form['Bank_Customer']
        if b_cust == "g":
            b_cust = 0.0
        elif b_cust == "gg":
            b_cust = 2.0
        else:
            b_cust = 1.0

        # Fetch value of Employment Status
        e_status = request.form['E_Status']
        if e_status == "t":
            e_status = 1.0
        else:
            e_status = 0.0

        # Fetch value for Credit Score
        cre_score = float(request.form['Credit_Score'])


        # Fetch value for Education Level
        e_level = request.form['Edu_Level']
        if e_level == "w":
            e_level = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "q":
            e_level = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "m":
            e_level = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "r":
            e_level = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "cc":
            e_level = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "k":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "c":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "d":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "x":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "i":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ,0.0, 0.0, 0.0]
        elif e_level == "e":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ,0.0, 0.0, 0.0]
        elif e_level == "aa":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,1.0, 0.0, 0.0]
        elif e_level == "ff":
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 1.0, 0.0]
        else:
            e_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 1.0]


        # Featch value for Prior Default
        default = request.form['Default']
        if default == "1":
            default = 0.0
        else:
            default = 1.0

        # Fetch value for Marital Status
        m_status = request.form['Mar_Status']
        if m_status == "Single":
            m_status = [1.0, 0.0, 0.0]
        elif m_status == "Married":
            m_status = [0.0, 1.0, 0.0]
        else:
            m_status = [0.0, 0.0, 1.0]

        # Fetch value for Citizenship
        citizen = request.form['Citizen']
        if citizen == "Yes":
            citizen = [1.0, 0.0, 0.0]
        elif citizen == "No":
            citizen = [0.0, 1.0, 0.0]
        else:
            citizen = [0.0, 0.0, 1.0]

        # Fetch value for Drivers Licence
        d_licence = request.form['D_Licence']
        if d_licence == "Yes":
            d_licence = [1.0, 0.0]
        else:
            d_licence = [0.0, 1.0]

        # Fetch value for Ethncity
        ethnicity = request.form['Ethnicity']
        if ethnicity == 'v':
            ethnicity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif ethnicity == 'h':
            ethnicity = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif ethnicity == 'bb':
            ethnicity = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif ethnicity == 'ff':
            ethnicity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif ethnicity == 'j':
            ethnicity = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        elif ethnicity == 'z':
            ethnicity = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        elif ethnicity == 'o':
            ethnicity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif ethnicity == 'dd':
            ethnicity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        else:
            ethnicity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


        #newX1 = [gender, age, debt, b_cust, experience, default, e_status, cre_score, income] + m_status + e_level + citizen + d_licence + ethnicity
        newX1 = [1.0, 0.2593985 , 0.07446429, 0.0, 0.00298246,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        newX2 = [1.0, 0.39849624, 0.76785714, 0.0, 0.70175439, 1.0, 1.0, 0.1641791, 0.024, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        if default == 0.0:
            newX = [newX2]
        else:
            newX = [newX1]

        X_nparray = np.asarray(newX, dtype=np.float32)

        my_prediction = rf.predict(X_nparray)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)