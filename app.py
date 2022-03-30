from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__) 
model = joblib.load(open('model.pkl', 'rb')) 

@app.route('/') 
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    intubated_free_days                     =   request.form['intubated_free_days']
    inotrope_free_days                      =   request.form['inotrope_free_days']
    length_of_stay_hours                    =   request.form['length_of_stay_hours']
    temp_celsius                            =   request.form['temp_celsius']
    onset_age_in_days                       =   request.form['onset_age_in_days']
    episode_id                              =   request.form['episode_id']
    time_to_antibiotics                     =   request.form['time_to_antibiotics']
    onset_hour_of_day                       =   request.form['onset_hour_of_day']
    race                                    =   request.form['race']
    unique_patient_id                       =   request.form['unique_patient_id']
    birth_weight_kg                         =   request.form['birth_weight_kg']
    inotrope_at_time_of_sepsis_eval         =   request.form['inotrope_at_time_of_sepsis_eval']
    gestational_age_at_birth_weeks          =   request.form['gestational_age_at_birth_weeks']
    sepsis_group                            =   request.form['sepsis_group']
    period                                  =   request.form['period']
    intubated_at_time_of_sepsis_evaluation  =   request.form['intubated_at_time_of_sepsis_evaluation']
    comorbidity_surgical                    =   request.form['comorbidity_surgical']
    sex                                     =   request.form['sex']
    blood_culture_positive                  =   request.form['blood_culture_positive']
    positive_days                           =   request.form['positive_days']
    umbilical_arterial_line                 =   request.form['umbilical_arterial_line']

    arr = np.array([[intubated_free_days, inotrope_free_days, length_of_stay_hours,temp_celsius,onset_age_in_days,episode_id,
        time_to_antibiotics,onset_hour_of_day,race,unique_patient_id,birth_weight_kg,inotrope_at_time_of_sepsis_eval,gestational_age_at_birth_weeks,
        sepsis_group,period,intubated_at_time_of_sepsis_evaluation,comorbidity_surgical,sex,blood_culture_positive,positive_days,
        umbilical_arterial_line]])


    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)