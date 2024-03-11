from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

model = pickle.load(open('models/flight_price_model.pkl','rb'))
# model_rfr = pickle.load(open('models/model_rfr.pkl','rb'))

df = pickle.load(open('models/flight_df.pkl','rb'))
airline_encoder = pickle.load(open('models/airline_encoder.pkl','rb'))
source_destination_encoder = pickle.load(open('models/source_destination_encoder.pkl','rb'))
stops_encoder = pickle.load(open('models/stops_encoder.pkl','rb'))
add_info_encoder = pickle.load(open('models/add_info_encoder.pkl','rb'))



airlines = df['Airline'].unique()
source = df['Source'].unique()
destination = df['Destination'].unique()
total_stops = df['Total_Stops'].unique()
addition_info =df['Additional_Info'].unique()
month = df['Month'].unique()


content = [airlines,source,destination,total_stops,addition_info,month]




app = Flask(__name__, static_folder='static')

@app.route('/')
def home():

    return render_template('index.html',content =content)

@app.route('/process', methods=['GET','POST'])
def process():
    day = int(request.form.get('day'))
    Month = int(request.form.get('Month'))
    hour = int(request.form.get('hour'))
    minutes = int(request.form.get('minutes'))
    airline = request.form.get('airline')
    source = request.form.get('source')
    destination = request.form.get('destination')
    total_stops = request.form.get('total_stops')
    addtional_info = request.form.get('addtional_info')
    
    total_duration=hour*60 + minutes

    day_month_duration = np.array([[day,Month,total_duration]])
    airline_array=airline_encoder.transform([[airline]]).toarray()
    source_destination_array=source_destination_encoder.transform([[source,destination]]).toarray()
    stops_array=stops_encoder.transform([[total_stops]]).toarray()
    add_info_array=add_info_encoder.transform([[addtional_info]]).toarray()
    test=np.concatenate((day_month_duration,airline_array,source_destination_array,stops_array,add_info_array),axis=1)
    result = model.predict(test)
    result=result[0]

    return render_template('index.html',result = result,content =content)





if __name__ == "__main__":
    app.run(debug=True)