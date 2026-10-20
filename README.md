# ⚡ Energy Forecasting with LSTM | ML Engineering & MLOps

An end-to-end machine learning project for hourly electricity consumption forecasting using a Long Short-Term Memory (LSTM) neural network built with PyTorch.

The project covers time-series preprocessing, model training, chronological evaluation, baseline comparison, REST API development with FastAPI, automated testing, and Docker-based packaging.

## 📌 Project Overview

Accurate electricity demand forecasting can help energy providers improve operational planning, allocate resources, and anticipate changes in consumption.

This project uses historical hourly electricity consumption data to predict the next hour's demand based on the previous 24 hours.

The goal is to demonstrate a practical machine learning workflow that combines deep learning, time-series forecasting, API development, testing, and deployment-oriented engineering practices.

## 🎯 Key Features

- Real-world electricity consumption dataset.
- Time-series preprocessing and chronological data splitting.
- LSTM-based forecasting model implemented in PyTorch.
- Training-only feature scaling to reduce data leakage.
- Validation-based model selection.
- Comparison against a seasonal naive baseline.
- REST API built with FastAPI.
- Input validation using Pydantic.
- Automated tests with pytest.
- Docker configuration for containerized execution.
- GitHub Actions workflow for automated testing.

## 📊 Dataset

**Dataset:** [Hourly Energy Consumption — AEP Hourly](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

The dataset contains historical hourly electricity consumption measurements for the AEP region.

| Property | Description |
|---|---|
| Source | Hourly Energy Consumption dataset |
| Region | AEP |
| Target variable | Electricity consumption in MW |
| Timestamp column | `Datetime` |
| Target column | `AEP_MW` |
| Approximate number of records | 121,000 |
| Forecast horizon | 1 hour |
| Input sequence length | 24 hours |

The data is sorted chronologically, and duplicate timestamps are handled during preprocessing.

The target variable is renamed to `consumption_mw`, and the timestamp column is converted to a datetime representation.

## 🧠 Machine Learning Approach

### Problem Formulation

This is a univariate time-series forecasting problem.

Given the previous 24 hourly consumption measurements, the model predicts the consumption for the next hour.

For a time series:

\[
x_1, x_2, \ldots, x_t
\]

the model uses the most recent 24 observations to predict:

\[
\hat{x}_{t+1} = f(x_{t-23}, \ldots, x_t)
\]

where:

- \(x_t\) represents the observed electricity consumption at time \(t\).
- \(\hat{x}_{t+1}\) represents the predicted consumption for the next hour.
- \(f\) represents the trained LSTM model.

### Data Preprocessing

The preprocessing pipeline includes:

1. Loading the historical electricity consumption data.
2. Converting timestamps to datetime values.
3. Sorting observations chronologically.
4. Handling duplicate timestamps.
5. Splitting the dataset into training, validation, and test sets.
6. Scaling the target variable using `MinMaxScaler`.
7. Creating sliding windows containing 24 historical observations.

The dataset is split chronologically:

| Split | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

Random splitting is avoided because it can introduce temporal leakage and produce unrealistic evaluation results.

The scaler is fitted on the training data only, then applied to the validation and test data.

The validation and test sequences use historical observations immediately preceding their respective prediction windows as context. This allows the model to use the available history without randomly mixing future observations into the training set.

### Model Architecture

The forecasting model uses an LSTM network implemented with PyTorch.

The model consists of:

1. An LSTM layer that learns temporal patterns from sequential observations.
2. A fully connected layer that maps the LSTM representation to a single numerical prediction.

The input contains 24 consecutive hourly measurements, and the output is a single predicted consumption value.

LSTMs are designed to learn dependencies in sequential data and can capture patterns that are difficult to represent using a simple feed-forward neural network.

### Training Strategy

The training pipeline includes:

- Mini-batch training with PyTorch `DataLoader`.
- Mean Squared Error (MSE) loss.
- Adam optimizer.
- Validation loss monitoring.
- Saving the model with the best validation performance.

The current training configuration uses a batch size of 32.

Model selection is based on validation performance rather than test-set performance. The test set is reserved for final evaluation.

## 📈 Results

The following results are documented from the project's evaluation runs.

### LSTM Performance

| Metric | Result |
|---|---:|
| MAE | 144.45 MW |
| RMSE | 192.62 MW |
| R² | 0.9940 |
| MAPE | 1.00% |

### Baseline Comparison

The model is compared against a seasonal naive baseline that predicts the next hour's consumption using the consumption observed at the same hour on the previous day.

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal naive baseline | 925.27 MW | 1,220.81 MW |
| LSTM | 144.45 MW | 192.62 MW |

The LSTM achieved an approximately **84.39% reduction in MAE** compared with the seasonal naive baseline.

### Interpretation

The reported metrics indicate that the LSTM performed substantially better than the selected baseline on the evaluated test period.

However, forecasting performance can vary with the evaluation period, data preprocessing, training configuration, and random initialization.

**Reproducibility note:** These metrics are documented from previous evaluation runs. The standalone evaluation script still needs to be completed and validated so that the results can be independently reproduced from the repository.

## 🚀 REST API

The project provides a REST API implemented with FastAPI.

The API loads the trained model and fitted scaler and exposes endpoints for health checks and hourly consumption forecasting.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Basic API information, if implemented |
| GET | `/health` | Reports API and model-loading status |
| POST | `/predict` | Generates a forecast from 24 historical observations |

The exact endpoints available depend on the current implementation in `api/main.py`.

### Prediction Input

The prediction endpoint expects a sequence containing 24 historical consumption values.

Example request:

```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      0.50, 0.52, 0.49, 0.48, 0.51, 0.54,
      0.56, 0.55, 0.53, 0.52, 0.50, 0.49,
      0.48, 0.47, 0.49, 0.51, 0.54, 0.57,
      0.59, 0.58, 0.56, 0.54, 0.52, 0.51
    ]
  }'
```

**Note:** This example assumes that the API expects scaled input values. The exact input and output contract should be confirmed against `api/schemas.py` and `api/main.py`.

### Example Response

The response contains the predicted consumption value according to the response schema implemented by the API.

The precise JSON field names and units should be taken from the current API implementation rather than assumed from this example.

### API Documentation

Once the API is running, FastAPI's interactive documentation is available at:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Visualization | Matplotlib |
| Machine learning | scikit-learn |
| Deep learning | PyTorch |
| API development | FastAPI, Uvicorn |
| Data validation | Pydantic |
| Testing | pytest |
| Containerization | Docker |
| Automation | GitHub Actions |

## 📁 Project Structure

```text
energy-forecasting-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
│   └── energy_consumption.csv
├── models/
│   ├── energy_lstm_model.pth
│   └── scaler.pkl
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## ⚙️ Installation

### Prerequisites

- Python 3.12
- pip
- Git

Docker is optional if you want to run the application in a container.

### 1. Clone the Repository

```bash
git clone https://github.com/RasulAramian/energy-forecasting-mlops.git
cd energy-forecasting-mlops
```

### 2. Create a Virtual Environment

```bash
python3.12 -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

The API should be accessible at:

```text
http://localhost:8001
```

Interactive documentation:

```text
http://localhost:8001/docs
```

**Important:** The trained model and scaler must be available at the paths expected by the application. If the model artifacts are not included in the repository, train the model first or obtain the artifacts through the project's documented process.

## 🐳 Docker

The repository includes a Dockerfile for packaging the API and its dependencies into a container.

### Build the Image

```bash
docker build -t energy-forecasting-mlops .
```

### Run the Container

```bash
docker run --rm -p 8001:8001 --name energy-forecasting-api energy-forecasting-mlops
```

Access the API at:

```text
http://localhost:8001
```

Interactive documentation:

```text
http://localhost:8001/docs
```

**Validation status:** The Docker workflow should be tested on a clean environment before the project is described as Docker-validated. The model artifacts must also be present inside the image or supplied through a documented volume or artifact-loading mechanism.

## 🧪 Testing

The project uses pytest for automated tests.

Run the test suite with:

```bash
pytest -v
```

The tests cover API behavior and model functionality.

Areas to validate include:

- API health checks.
- Input validation for sequences of the wrong length.
- Successful predictions when the model and scaler are available.
- Expected error responses when model artifacts are unavailable.
- Model output shape and numerical validity.

Tests for missing model artifacts should explicitly simulate the unavailable-model condition and assert the expected HTTP status code. A test that only checks a status code when it happens to equal a particular value does not reliably verify the intended behavior.

## 🔄 MLOps and Engineering Practices

The project demonstrates several practices used in practical machine learning systems:

- **Reproducible data processing:** Explicit preprocessing and sequence construction.
- **Chronological evaluation:** Training, validation, and test sets are separated by time.
- **Leakage prevention:** The scaler is fitted only on the training data.
- **Model selection:** Validation loss is used to select the saved model.
- **Model serving:** FastAPI provides an interface for inference.
- **Input validation:** Pydantic schemas define the request structure.
- **Containerization:** Docker packages the application and its dependencies.
- **Automated testing:** pytest checks application and model behavior.
- **Continuous integration:** GitHub Actions runs the configured test workflow.

### Current Limitations

Some engineering tasks still require validation or completion:

- Complete and validate the standalone evaluation script.
- Reproduce the reported test metrics from the repository.
- Strengthen tests for missing or invalid model artifacts.
- Verify the Docker build and runtime behavior.
- Confirm that the GitHub Actions workflow passes on the latest commit.
- Add monitoring and a documented deployment process.

These items should be considered pending until they have been implemented and verified.

## 🔮 Future Improvements

Potential improvements include:

- Completing a reproducible evaluation pipeline.
- Adding prediction and actual-value plots.
- Comparing LSTM with GRU, linear models, and other forecasting baselines.
- Adding rolling-window backtesting.
- Evaluating performance across different seasons and demand regimes.
- Improving API error handling and readiness checks.
- Adding structured logging and inference monitoring.
- Publishing a validated deployment.
- Tracking experiments and model versions.
- Automating model training and artifact management.

## 📌 Key Takeaways

This project brings together time-series forecasting and practical machine learning engineering.

It demonstrates experience with:

- Preparing real-world time-series data.
- Building and training an LSTM model with PyTorch.
- Evaluating forecasts against a baseline.
- Exposing a trained model through a REST API.
- Writing automated tests.
- Preparing a project for containerized execution and continuous integration.

The reported evaluation results are promising, while full metric reproducibility, Docker validation, and several production-readiness improvements remain important next steps.

## 👨‍💻 Author

**Rasul Aramian**

PhD in Mathematics

Applying mathematical and programming skills to machine learning, deep learning, and practical ML engineering projects.

- **GitHub:** [RasulAramian](https://github.com/RasulAramian)
- **Project Repository:** [energy-forecasting-mlops](https://github.com/RasulAramian/energy-forecasting-mlops)
