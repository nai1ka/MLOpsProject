<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
![Test code workflow](https://github.com/nai1ka/MLOpsProject/actions/workflows/test-code.yaml/badge.svg)
![Validate model workflow](https://github.com/nai1ka/MLOpsProject/actions/workflows/validate-model.yaml/badge.svg)


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/nai1ka/MLOpsProject">
    <img src="logo.svg" alt="Logo" width="200">
  </a>

<h3 align="center">Taxi Fare Price Prediction</h3>

 
</div>



<!-- ABOUT THE PROJECT -->
## About The Project
Unfair taxi prices can lead to customer dissatisfaction and affect profitability. Taxi companies aim to balance revenue and customer expectations. A machine learning model can be built to predict fair taxi prices based on time, location, and weather conditions, maximizing revenue and improving customer satisfaction

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



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Installation


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
   ```
3. Deploy the model in a docker container
   ```sh
   docker build -t predict_taxi_price_ml_service api
   docker run --rm -p 5151:8080 predict_taxi_price_ml_service -d
   ```
3. Launch Flask using `app.py`
   ```sh
   python3 src/app.py
   ```
4. Run Gradio web UI
   ```sh
   python3 src/ui.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

The project can be used to predict taxi prices based on time, location, and weather conditions. The model can be deployed in a docker container and accessed via a web UI.
<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributors


<a href="https://github.com/nai1ka"><img src="https://avatars.githubusercontent.com/u/40440192?v=4" title="nai1ka" width="80" height="80"></a>
<a href="https://github.com/arinagoncharova2005"><img src="https://avatars.githubusercontent.com/u/71409384?v=4" title="arinagoncharova2005" width="80" height="80"></a>
<a href="https://github.com/Zaurall"><img src="https://avatars.githubusercontent.com/u/117632304?v=4" title="Zaurall" width="80" height="80"></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

