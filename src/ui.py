# src/ui.py

import gradio as gr
import mlflow
from hydra import compose, initialize
from omegaconf import OmegaConf
from data import transform_data
import json
import requests
import numpy as np
import pandas as pd

initialize(version_base=None, config_path="../configs")
cfg = compose(config_name="main")

def predict(apparentTemperature = None,
            distance = None,
            hour = None,
            month = None,
            day = None,
            surge_multiplier = None,
            precipIntensity = None,
            precipProbability = None,
            humidity = None,
            windSpeed = None,
            visibility = None,
            pressure = None,
            windBearing = None,
            cloudCover = None,
            uvIndex = None,
            precipIntensityMax = None,
            source = None,
            destination = None,
            name = None,
            short_summary = None):
    
    datetime_df = pd.DataFrame({
                   'year': [2018],
                   'month': [month],
                   'day': [day],
                   "hour": [hour],
                   })
    features = {
            "apparentTemperature" : apparentTemperature,
            "distance" : distance,
            "hour" : hour,
            "month" : month,
            "day" : day,
            "surge_multiplier" : surge_multiplier,
            "precipIntensity" : precipIntensity,
            "precipProbability" : precipProbability,
            "humidity" : humidity,
            "windSpeed" : windSpeed,
            "visibility" : visibility,
            "pressure" : pressure,
            "windBearing" : windBearing,
            "cloudCover" : cloudCover,
            "uvIndex" : uvIndex,
            "precipIntensityMax" : precipIntensityMax,
            "source" : source,
            "destination" : destination,
            "name" : name,
            "short_summary" : short_summary,
            "datetime": pd.to_datetime(datetime_df)
    }
    
    
    raw_df = pd.DataFrame(features, index=[0])
    
    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    X = transform_data(
                        df = raw_df, 
                        cfg = cfg, 
                        return_df = False, 
                        only_X = True,
                        transformer_version="v1",
                        only_transform=True
                      )
    
    # Convert it into JSON
    example = X.iloc[0,:]

    example = json.dumps( 
        { "inputs": example.to_dict() }
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:{cfg.flask_port}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    return response.json()['price']

# Only one interface is enough
demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict,
    
    # Here, the arguments in `predict` function
    # will populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        gr.Number(label="apparentTemperature", info = "in Fahrenheit"), 
        gr.Number(label="distance", info = "in miles"),
        gr.Number(label="hour"),
        gr.Number(label="month"),
        gr.Number(label="day"),
        gr.Number(label="surge_multiplier"),
        gr.Number(label="precipIntensity"),
        gr.Number(label="precipProbability"),
        gr.Number(label="humidity"),
        gr.Number(label="windSpeed"),
        gr.Number(label="visibility"),
        gr.Number(label="pressure"),
        gr.Number(label="windBearing"),
        gr.Number(label="cloudCover"),
        gr.Number(label="uvIndex"),
        gr.Number(label="precipIntensityMax"),
        gr.Dropdown(label="source", choices=['Haymarket Square', 'Back Bay', 'North End', 'North Station', 'Beacon Hill',
 'Boston University', 'Fenway', 'South Station', 'Theatre District',
 'West End', 'Financial District' ,'Northeastern University']), 
        gr.Dropdown(label="destination", choices=['North Station' ,'Northeastern University', 'West End' ,'Haymarket Square',
 'South Station' ,'Fenway', 'Theatre District', 'Beacon Hill', 'Back Bay'
 'North End', 'Financial District', 'Boston University']), 
        gr.Dropdown(label="name", choices=['Shared', 'Lux' ,'Lyft' ,'Lux Black XL', 'Lyft XL', 'Lux Black', 'UberXL',
 'Black', 'UberX', 'WAV', 'Black SUV' ,'UberPool', 'Taxi']), 
        gr.Dropdown(label="short_summary", choices=[' Mostly Cloudy ', ' Rain ' ,' Clear ' ,' Partly Cloudy ', ' Overcast ',
 ' Light Rain ' ,' Foggy ', ' Possible Drizzle ' ,' Drizzle ']),   
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="Taxi ride price prediction"),
    
    # This will provide the user with examples to test the API
    examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

# Launch the web UI locally on port 5155
demo.launch(server_port = 5155,server_name="0.0.0.0")
