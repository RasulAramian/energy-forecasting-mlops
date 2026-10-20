# ⚡ Energy Consumption Forecasting with LSTM & MLOps

### End-to-End Deep Learning for Hourly Electricity Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED)
![Machine Learning](https://img.shields.io/badge/Deep_Learning-Time_Series-purple)
![License](https://img.shields.io/badge/License-Not_Specified-lightgrey)

An end-to-end deep learning project for forecasting hourly electricity consumption using a **Long Short-Term Memory (LSTM)** neural network.

The project demonstrates a practical machine learning workflow, including time-series preprocessing, chronological validation, sequence generation, neural network training, performance evaluation, REST API development, automated testing, and containerization configuration.

The primary objective is to develop a reproducible forecasting system that can eventually be deployed as a production-oriented machine learning service.

> **Project Status:** Model training and evaluation completed. FastAPI implementation available. Docker validation, CI improvements, and cloud deployment are in progress.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Data Preprocessing](#-data-preprocessing)
- [LSTM Model Architecture](#-lstm-model-architecture)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [REST API](#-rest-api)
- [Installation](#-installation)
- [Docker](#-docker)
- [Testing](#-testing)
- [MLOps Components](#-mlops-components)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Project Overview

Accurate electricity demand forecasting is an important problem in energy management and power system operations.

Electricity consumption changes continuously due to factors such as daily activity patterns, seasonal variations, weather conditions, and economic activity.

Traditional forecasting approaches may struggle to capture complex temporal dependencies.

This project investigates the use of an LSTM neural network to learn patterns from historical electricity consumption and predict future demand.

### Project Objectives

1. Process and analyze a real-world electricity consumption dataset.
2. Prevent data leakage through chronological train-validation-test splitting.
3. Transform historical observations into supervised learning sequences.
4. Train an LSTM model using PyTorch.
5. Evaluate forecasting accuracy using multiple regression metrics.
6. Compare model performance against a seasonal naive baseline.
7. Expose the trained model through a FastAPI REST interface.
8. Prepare the application for Docker-based deployment and automated testing.

### Forecasting Task

The model uses the previous **24 hourly electricity consumption measurements** to predict consumption for the next hour.

**Input:**

```text
[t-23, t-22, t-21, ..., t-1, t]
```

**Output:**

```text
Predicted electricity consumption at t+1
```

This is a one-step-ahead forecasting task.

---

## 💼 Business Problem

Electricity demand forecasting supports several operational decisions:

- Power generation scheduling
- Energy resource allocation
- Grid load management
- Short-term operational planning
- Electricity demand monitoring

Improved forecasting accuracy can help energy operators make more informed decisions and reduce uncertainty.

This project focuses on the machine learning component of the forecasting process rather than a complete electricity grid management system.

---

## 📊 Dataset

### Source

**PJM Interconnection Hourly Energy Consumption Dataset**

The project uses the AEP hourly electricity load dataset.

Dataset source:

[Hourly Energy Consumption — Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

### Dataset Description

The original dataset contains approximately 121,000 hourly electricity load observations.

The relevant columns are:

| Column | Description |
|---|---|
| `Datetime` | Timestamp of the observation |
| `AEP_MW` | Electricity load measured in megawatts (MW) |

For consistency, the columns are renamed:

```text
Datetime → datetime
AEP_MW   → consumption_mw
```

### Important Note

The target variable represents electrical power demand in **MW**, not accumulated electrical energy in MWh.

The dataset contains historical measurements, allowing the forecasting problem to be formulated as univariate time-series regression.

---

## 🛠 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Data Processing | pandas, NumPy |
| Feature Scaling | scikit-learn |
| Data Visualization | Matplotlib |
| Model Architecture | LSTM |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Data Validation | Pydantic |
| Testing | pytest |
| Containerization | Docker |
| CI Configuration | GitHub Actions |
| Version Control | Git, GitHub |

---

## 🏗 Project Architecture

The project follows a modular structure separating machine learning components, API functionality, testing, and deployment configuration.

```text
energy-forecasting-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── energy_consumption.csv
│
├── models/
│   ├── energy_lstm_model.pth
│   └── scaler.pkl
│
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

**Note:** Dataset and model artifact files are generated or maintained locally and may not be included in the Git repository.

### Machine Learning Workflow

```text
              Real-World Electricity Data
                          |
                          v
                 Data Exploration
                          |
                          v
                Data Preprocessing
                          |
                          v
               Chronological Splitting
                          |
                 +--------+--------+
                 |        |        |
                 v        v        v
               Train   Validation  Test
                 |
                 v
           Feature Scaling
                 |
                 v
          Sequence Generation
                 |
                 v
             LSTM Model
                 |
                 v
          Model Optimization
                 |
                 v
        Validation & Evaluation
                 |
                 v
          Model Serialization
                 |
                 v
             FastAPI
                 |
                 v
          Docker Container
                 |
                 v
        Deployment (Planned)
```

---

## 🔄 Data Preprocessing

Time-series forecasting requires special care to avoid introducing information from the future into model training.

### 1. Data Cleaning

The preprocessing workflow includes:

- Parsing datetime values
- Sorting observations chronologically
- Identifying duplicate timestamps
- Removing duplicate observations
- Preparing the target variable

Four duplicate timestamps were removed during data preparation.

### 2. Chronological Data Splitting

Unlike ordinary machine learning problems, time-series observations should not be randomly shuffled before splitting.

The dataset is divided chronologically:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The training set contains the earliest observations, followed by validation and testing periods.

This approach provides a more realistic evaluation of forecasting performance.

### 3. Feature Scaling

A `MinMaxScaler` is fitted exclusively on the training data.

The fitted scaler is then applied to the validation and test sets.

This prevents information from future observations from influencing the scaling process.

### 4. Sequence Generation

A sliding-window approach transforms the original time series into supervised learning samples.

Each input sequence contains 24 consecutive hourly observations.

The target is the following hourly observation.

For example:

```text
Input sequence:
[x1, x2, x3, ..., x24]

Target:
x25
```

The model learns to predict the next value from the preceding 24 observations.

---

## 🧠 LSTM Model Architecture

The forecasting model is implemented using PyTorch.

Long Short-Term Memory networks are recurrent neural networks designed to capture temporal relationships in sequential data.

LSTM networks use gating mechanisms to regulate information flow through their internal states.

### Architecture

```text
Input Sequence
   |
   | Shape: (batch_size, 24, 1)
   |
   v
LSTM Layer
   |
   | Hidden Size: 50
   | Number of Layers: 1
   |
   v
Last Time-Step Hidden State
   |
   v
Fully Connected Layer
   |
   | 50 → 25
   |
   v
ReLU Activation
   |
   v
Fully Connected Layer
   |
   | 25 → 1
   |
   v
Predicted Electricity Load
```

### Model Configuration

| Parameter | Value |
|---|---|
| Model Type | LSTM |
| Input Features | 1 |
| Sequence Length | 24 |
| Hidden Units | 50 |
| LSTM Layers | 1 |
| Fully Connected Hidden Units | 25 |
| Output Features | 1 |
| Forecast Horizon | 1 hour |

The model processes a sequence of historical measurements and produces a single continuous prediction.

---

## 🚀 Model Training

The model was trained using PyTorch.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Mean Squared Error |
| Batch Size | 64 |
| Training Epochs | 10 |
| Sequence Length | 24 |

### Training Process

The training process consists of:

1. Loading training sequences in mini-batches.
2. Performing forward propagation through the LSTM.
3. Computing the Mean Squared Error loss.
4. Calculating gradients through backpropagation.
5. Updating model parameters using Adam.
6. Monitoring performance on the validation dataset.

The trained model is saved as:

```text
models/energy_lstm_model.pth
```

The fitted scaler is stored as:

```text
models/scaler.pkl
```

These artifacts are required for model inference.

---

## 📈 Model Evaluation

The trained LSTM was evaluated on a chronologically separated test dataset.

Predictions were transformed back to the original MW scale before computing evaluation metrics.

### LSTM Performance

| Metric | Test Result |
|---|---:|
| MAE | 144.45 MW |
| RMSE | 192.62 MW |
| R² Score | 0.9940 |
| MAPE | 1.00% |

### Metric Interpretation

**Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted electricity load.

**Root Mean Squared Error (RMSE)** penalizes larger prediction errors more strongly.

**R² Score** measures how much of the variation in the target variable is explained by the model.

**Mean Absolute Percentage Error (MAPE)** expresses prediction error as a percentage of actual values.

### Baseline Comparison

A seasonal naive forecasting approach was used as a reference.

The baseline predicts the electricity load at a given hour using the observed load from exactly 24 hours earlier.

```text
Prediction at time t = Actual load at time t - 24 hours
```

This provides a meaningful benchmark for hourly electricity consumption forecasting.

### Verified Comparison

The final comparison was performed using time-aligned predictions.

| Metric | Seasonal Naive Baseline | LSTM |
|---|---:|---:|
| MAE | 925.27 MW | 144.45 MW |
| RMSE | 1220.81 MW | 192.62 MW |

**Evaluation Details:**

- Total test predictions: 18,167
- Skipped samples: 2
- Aligned evaluation samples: 18,165

### Key Result

The LSTM achieved an **84.39% reduction in Mean Absolute Error** compared with the 24-hour seasonal naive baseline.

This indicates that the trained model captured useful temporal patterns beyond simply repeating the load observed on the previous day.

The reported improvement is relative to the selected baseline on the evaluated test period. It does not guarantee equivalent performance on future or previously unseen operating conditions.

---

## 🌐 REST API

A REST API is implemented using FastAPI to make the trained model accessible through HTTP requests.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Returns the API health status |
| POST | `/predict` | Predicts the next hourly electricity load |
| GET | `/docs` | Interactive Swagger API documentation |
| GET | `/openapi.json` | OpenAPI specification |

### Prediction Request

**Endpoint:**

```http
POST /predict
```

**Example JSON Request:**

```json
{
  "sequence": [
    15000, 15100, 14950, 14800,
    14700, 14650, 14500, 14400,
    14600, 14900, 15300, 15700,
    16000, 16200, 16400, 16300,
    16100, 15900, 15800, 15700,
    15600, 15500, 15400, 15300
  ]
}
```

The request must contain exactly 24 numeric values representing consecutive hourly electricity load measurements in MW.

### Example Response Structure

```json
{
  "predicted_value": 15250.75,
  "status": "success"
}
```

The value above is illustrative and does not represent a verified prediction from the trained model.

### Input Validation

The API uses Pydantic to validate incoming requests.

Invalid input lengths are rejected, helping prevent malformed sequences from reaching the prediction pipeline.

### Interactive Documentation

After starting the application, Swagger documentation is available at:

```text
http://localhost:8001/docs
```

---

## 💻 Installation

### Prerequisites

- Python 3.12
- Git
- pip
- Sufficient memory for PyTorch and model dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/RasulAramian/energy-forecasting-mlops.git
cd energy-forecasting-mlops
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate the Environment

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare Model Artifacts

The API requires the following files:

```text
models/energy_lstm_model.pth
models/scaler.pkl
```

Ensure both files are available before starting the prediction service.

The training notebook and project source files document the model development workflow.

### 6. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Open:

```text
http://localhost:8001/docs
```

Use the interactive interface to submit prediction requests.

---

## 🐳 Docker

The repository includes a Dockerfile for packaging the application and its dependencies.

### Build the Docker Image

```bash
docker build -t energy-forecasting-api .
```

### Run the Container

```bash
docker run -d \
  --name energy-api-container \
  -p 8001:8001 \
  energy-forecasting-api
```

### Access the API

```text
http://localhost:8001/docs
```

### Important

The required model and scaler artifacts must be accessible inside the container.

The Docker build and runtime configuration are pending final validation.

---

## 🧪 Testing

The project includes automated tests implemented using pytest.

The testing structure includes:

```text
tests/
├── test_api.py
└── test_model.py
```

### Run Tests

```bash
pytest -v
```

The test suite covers API and model-related functionality.

Additional inference and artifact-loading checks are planned to strengthen reliability.

---

## ⚙️ MLOps Components

The project incorporates several foundational MLOps concepts.

### 1. Modular Project Structure

Model architecture, data processing, API functionality, and tests are organized into separate modules.

This improves maintainability and supports future development.

### 2. Model Serialization

The trained PyTorch model is saved as a `.pth` artifact.

The fitted preprocessing scaler is serialized separately.

This enables the same preprocessing transformation to be reused during inference.

### 3. API-Based Inference

FastAPI exposes the model through an HTTP interface.

This separates model inference from the training environment and allows other applications to consume predictions.

### 4. Containerization

A Dockerfile defines the application environment and dependencies.

Containerization helps make the inference service more portable across compatible systems.

### 5. Automated Testing

pytest provides a foundation for verifying application behavior and detecting regressions.

### 6. Continuous Integration

The repository contains a GitHub Actions workflow intended to automate dependency installation and testing.

The workflow is undergoing improvements to avoid unnecessary model retraining and to better handle external artifacts.

### Current MLOps Status

| Component | Status |
|---|---|
| Data Preprocessing | Completed |
| Chronological Validation | Completed |
| LSTM Training | Completed |
| Model Evaluation | Completed |
| Baseline Comparison | Completed |
| Model Serialization | Implemented |
| FastAPI | Implemented |
| Automated Tests | Initial Implementation |
| Docker Configuration | Created — Validation Pending |
| CI Workflow | Created — Improvements Pending |
| Cloud Deployment | Planned |
| Production Monitoring | Not Implemented |

---

## 🔮 Future Improvements

Several enhancements could improve the forecasting system.

### Model Improvements

- Evaluate longer historical input windows.
- Explore multivariate forecasting using weather and calendar features.
- Compare LSTM with alternative forecasting architectures.
- Evaluate model performance across different seasonal periods.

### Software Engineering

- Improve model-loading error handling.
- Add successful inference tests.
- Improve dependency management.
- Strengthen artifact validation.

### MLOps

- Validate Docker image builds and container execution.
- Improve GitHub Actions workflows.
- Deploy the API to a cloud hosting platform.
- Add structured logging.
- Introduce basic model and service monitoring.

These improvements are planned extensions rather than features already available in the current implementation.

---

## 🎓 Key Learning Outcomes

This project demonstrates practical experience with:

- Real-world time-series data processing
- Chronological train-validation-test splitting
- Data leakage prevention
- Sliding-window sequence generation
- LSTM neural networks
- PyTorch training loops
- Regression evaluation metrics
- Seasonal naive baseline comparison
- Model serialization
- FastAPI development
- REST API request validation
- Automated testing
- Docker configuration
- Foundational MLOps practices

The project connects deep learning experimentation with the engineering components required to serve predictions through an API.

---

## 👨‍💻 Author

**Rasul Aramian**

PhD in Mathematics | Machine Learning & Deep Learning

Interested in machine learning engineering, deep learning, mathematical modeling, and production-oriented AI systems.

**GitHub:** [RasulAramian](https://github.com/RasulAramian)

**Project Repository:** [energy-forecasting-mlops](https://github.com/RasulAramian/energy-forecasting-mlops)

---

⭐ If you find this project useful, consider starring the repository.
