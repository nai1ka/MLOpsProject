![Test code workflow](https://github.com/nai1ka/MLOpsProject/actions/workflows/test-code.yaml/badge.svg)
![Validate model workflow](https://github.com/nai1ka/MLOpsProject/actions/workflows/validate-model.yaml/badge.svg)


<br />
<div align="center">
  <a href="https://github.com/nai1ka/MLOpsProject">
    <img src="materials/logo.svg" alt="Logo" width="200">
  </a>

<h3 align="center">Taxi Fare Price Prediction</h3>
</div>



## About The Project

Unfair taxi prices can lead to customer dissatisfaction and affect profitability. Taxi companies aim to balance revenue
and customer expectations. A machine learning model can be built to predict fair taxi prices based on time, location,
and weather conditions, maximizing revenue and improving customer satisfaction

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Getting Started

## Deploy Using Docker

### Using Image from Docker Hub

1. Pull the image from Docker Hub
   ```sh
   docker pull nai1ka/predict_taxi_price_ml_service
   docker run --rm -p 5151:8080 nai1ka/predict_taxi_price_ml_service -d
   ```

### Building the Image Locally

1. Clone the repo
   ```sh
    git clone https://github.com/nai1ka/MLOpsProject.git
    cd MLOpsProject
    ```
2. Build the image
    ```sh
    docker build -t predict_taxi_price_ml_service api
    docker run --rm -p 5151:8080 predict_taxi_price_ml_service -d
    ```
3. Access the API at `http://localhost:5151/`

## Deploy Using Flask

1. Clone the repo
   ```sh
    git clone https://github.com/nai1ka/MLOpsProject.git
    cd MLOpsProject
    ```
2. Install pip packages
    ```sh
      pip install -r requirements.txt
    ```
3. Configure the environment variables
    ```sh
    export PROJECTPATH=$PWD
    export PYTHONPATH=$PWD/src
    ```
4. Launch Flask using `app.py`
    ```sh
    python3 src/app.py
    ```
5. Access the API at `http://localhost:8080/`

6.  Run Gradio UI (optional)
   ```sh
   python3 src/ui.py
   ````
   ![UI demo](materials/ui_demo.mp4)

## Examples of API Requests

### Request to model in Docker Container

```sh
curl -X POST localhost:5151/invocations \
     -H 'Content-Type: application/json' \
     -d '{"inputs": {"apparentTemperature": 0.4612996229762697, "cloudCover": 0.03, "day": 0.9310344827586207, "day_cos": 0.8207634412072763, "day_of_week": 0.3333333333333333, "day_of_week_cos": -0.22252093395631437, "day_of_week_sin": 0.9749279121818238, "day_sin": -0.5712682150947923, "destination_Back Bay": 0.0, "destination_Beacon Hill": 0.0, "destination_Boston University": 0.0, "destination_Fenway": 0.0, "destination_Financial District": 0.0, "destination_Haymarket Square": 0.0, "destination_North End": 0.0, "destination_North Station": 1.0, "destination_Northeastern University": 0.0, "destination_South Station": 0.0, "destination_Theatre District": 0.0, "destination_West End": 0.0, "distance": 0.0, "hour": 0.043478260869565216, "hour_cos": 0.9659258262890684, "hour_sin": 0.2588190451025208, "humidity": 0.5434782608695652, "month": 0.0, "month_cos": 0.8660254037844383, "month_sin": -0.5000000000000003, "name_Black": 0.0, "name_Black SUV": 0.0, "name_Lux": 0.0, "name_Lux Black": 0.0, "name_Lux Black XL": 0.0, "name_Lyft": 1.0, "name_Lyft XL": 0.0, "name_Shared": 0.0, "name_Taxi": 0.0, "name_UberPool": 0.0, "name_UberX": 0.0, "name_UberXL": 0.0, "name_WAV": 0.0, "precipIntensity": 0.0, "precipIntensityMax": 0.7399165507649512, "precipProbability": 0.0, "pressure": 0.08920587609112118, "short_summary_ Clear ": 1.0, "short_summary_ Drizzle ": 0.0, "short_summary_ Foggy ": 0.0, "short_summary_ Light Rain ": 0.0, "short_summary_ Mostly Cloudy ": 0.0, "short_summary_ Overcast ": 0.0, "short_summary_ Partly Cloudy ": 0.0, "short_summary_ Possible Drizzle ": 0.0, "short_summary_ Rain ": 0.0, "source_Back Bay": 0.0, "source_Beacon Hill": 0.0, "source_Boston University": 0.0, "source_Fenway": 0.0, "source_Financial District": 0.0, "source_Haymarket Square": 1.0, "source_North End": 0.0, "source_North Station": 0.0, "source_Northeastern University": 0.0, "source_South Station": 0.0, "source_Theatre District": 0.0, "source_West End": 0.0, "surge_multiplier": 0.0, "uvIndex": 0.0, "visibility": 1.0, "windBearing": 0.672316384180791, "windSpeed": 0.46993780234968907}}'
````

### Request to model in Flask

```sh
 curl -X POST localhost:5001/predict\
      -H 'Content-Type: application/json' \
      -d '{"inputs": {"apparentTemperature": 0.4612996229762697, "cloudCover": 0.03, "day": 0.9310344827586207, "day_cos": 0.8207634412072763, "day_of_week": 0.3333333333333333, "day_of_week_cos": -0.22252093395631437, "day_of_week_sin": 0.9749279121818238, "day_sin": -0.5712682150947923, "destination_Back Bay": 0.0, "destination_Beacon Hill": 0.0, "destination_Boston University": 0.0, "destination_Fenway": 0.0, "destination_Financial District": 0.0, "destination_Haymarket Square": 0.0, "destination_North End": 0.0, "destination_North Station": 1.0, "destination_Northeastern University": 0.0, "destination_South Station": 0.0, "destination_Theatre District": 0.0, "destination_West End": 0.0, "distance": 0.0, "hour": 0.043478260869565216, "hour_cos": 0.9659258262890684, "hour_sin": 0.2588190451025208, "humidity": 0.5434782608695652, "month": 0.0, "month_cos": 0.8660254037844383, "month_sin": -0.5000000000000003, "name_Black": 0.0, "name_Black SUV": 0.0, "name_Lux": 0.0, "name_Lux Black": 0.0, "name_Lux Black XL": 0.0, "name_Lyft": 1.0, "name_Lyft XL": 0.0, "name_Shared": 0.0, "name_Taxi": 0.0, "name_UberPool": 0.0, "name_UberX": 0.0, "name_UberXL": 0.0, "name_WAV": 0.0, "precipIntensity": 0.0, "precipIntensityMax": 0.7399165507649512, "precipProbability": 0.0, "pressure": 0.08920587609112118, "short_summary_ Clear ": 1.0, "short_summary_ Drizzle ": 0.0, "short_summary_ Foggy ": 0.0, "short_summary_ Light Rain ": 0.0, "short_summary_ Mostly Cloudy ": 0.0, "short_summary_ Overcast ": 0.0, "short_summary_ Partly Cloudy ": 0.0, "short_summary_ Possible Drizzle ": 0.0, "short_summary_ Rain ": 0.0, "source_Back Bay": 0.0, "source_Beacon Hill": 0.0, "source_Boston University": 0.0, "source_Fenway": 0.0, "source_Financial District": 0.0, "source_Haymarket Square": 1.0, "source_North End": 0.0, "source_North Station": 0.0, "source_Northeastern University": 0.0, "source_South Station": 0.0, "source_Theatre District": 0.0, "source_West End": 0.0, "surge_multiplier": 0.0, "uvIndex": 0.0, "visibility": 1.0, "windBearing": 0.672316384180791, "windSpeed": 0.46993780234968907}}' \
      
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* ![Airflow](https://img.shields.io/badge/Airflow-v2.7.3-blue?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)
* ![Mlflow](https://img.shields.io/badge/MLFlow-v2.14.1-blue?style=for-the-badge&logo=mlflow&logoColor=61DAFB)
* ![Airflow](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
* ![ZenML](https://img.shields.io/badge/ZENML-ae7bdb?style=for-the-badge)
* ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
* ![Hydra](https://img.shields.io/badge/Hydra-7bbac7?style=for-the-badge&logoColor=white)
* ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
* ![SKLearn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
* ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
* ![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributors

<a href="https://github.com/nai1ka"><img src="https://avatars.githubusercontent.com/u/40440192?v=4" title="nai1ka" width="80" height="80"></a>
<a href="https://github.com/arinagoncharova2005"><img src="https://avatars.githubusercontent.com/u/71409384?v=4" title="arinagoncharova2005" width="80" height="80"></a>
<a href="https://github.com/Zaurall"><img src="https://avatars.githubusercontent.com/u/117632304?v=4" title="Zaurall" width="80" height="80"></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


