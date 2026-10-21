# Energy Forecasting with LSTM \| ML Engineering & MLOps

An end-to-end machine learning project for hourly electricity
consumption forecasting using a PyTorch Long Short-Term Memory (LSTM)
network.

The project covers time-series preprocessing, model training,
evaluation, a FastAPI prediction service, automated tests, Docker
packaging, and deployment on Render.

## Project Overview

The model uses the previous 24 hourly electricity consumption
observations to predict consumption for the next hour.

## Key Features

- Real-world hourly electricity consumption data from the AEP region.
- Chronological train, validation, and test splits.
- LSTM forecasting model implemented with PyTorch.
- `MinMaxScaler` fitted on training data only.
- Model selection based on validation loss.
- FastAPI prediction endpoint with Pydantic request validation.
- Automated tests with pytest.
- Docker configuration and GitHub Actions workflow.
- Render deployment.

## Dataset

**Dataset:** [Hourly Energy Consumption — AEP
Hourly](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

The project CSV uses `Datetime` as the timestamp column and `AEP_MW` as
the target column.

| Property              | Description             |
|-----------------------|-------------------------|
| Region                | AEP                     |
| Target                | Electricity consumption |
| Target unit           | Megawatts (MW)          |
| Forecast horizon      | 1 hour                  |
| Input sequence length | 24 observations         |

The preprocessing code parses timestamps, sorts rows chronologically,
and drops rows with missing target values. It does not currently remove
duplicate timestamps, so duplicate handling should be reviewed if the
dataset changes.

## Machine Learning Approach

### Problem Formulation

This is a univariate time-series forecasting problem. Given the most
recent 24 hourly consumption measurements, the model predicts the next
hourly value:

Given the previous 24 hourly observations, the model predicts the next hourly consumption value:

`Prediction at time t+1 = f(observations from time t-23 through time t)`

### Preprocessing

The current pipeline:

1.  Loads the CSV file and parses timestamps.
2.  Sorts observations chronologically and drops rows with missing
    target values.
3.  Splits observations chronologically into training, validation, and
    test portions.
4.  Fits `MinMaxScaler` on the training portion only.
5.  Applies the scaler to each portion.
6.  Creates sliding windows containing 24 observations.

| Split      | Approximate proportion |
|------------|-----------------------:|
| Training   |                    70% |
| Validation |                    15% |
| Test       |                    15% |

The validation and test portions include the preceding 24 observations
as context for sequence construction. Random splitting is avoided to
preserve time order.

### Model Architecture

The model in `src/model.py` consists of an LSTM layer, a fully connected
layer with 25 units, a ReLU activation, and a final fully connected
layer that produces one prediction. The input shape is
`(batch_size, 24, 1)` and the output shape is `(batch_size, 1)`.

### Training

The current training configuration uses a batch size of 32, Mean Squared
Error loss, Adam with a learning rate of 0.001, and five epochs. The
saved model is selected using validation loss. Model weights and the
fitted scaler are stored in `models/`.

## Evaluation Results

The latest local execution of `python -m src.evaluate` produced these
results for the saved model and current test preprocessing:

| Metric                         |      Result |
|--------------------------------|------------:|
| Mean Absolute Error (MAE)      | 154.3809 MW |
| Root Mean Squared Error (RMSE) | 205.9744 MW |

Previously documented metrics were not reproduced by the current run and
are not presented here as verified results. Results may vary if the
dataset, saved model, preprocessing, dependencies, or training
configuration changes.

To run evaluation locally:

``` bash
python -m src.evaluate
```

## REST API

The API is built with FastAPI. At startup, it loads the saved model and
scaler.

### Deployment

- **API:** https://energy-forecasting-mlops.onrender.com
- **Interactive documentation:**
  https://energy-forecasting-mlops.onrender.com/docs
- **Health check:** https://energy-forecasting-mlops.onrender.com/health

### Endpoints

| Method | Endpoint   | Description                                                    |
|--------|------------|----------------------------------------------------------------|
| `GET`  | `/health`  | Reports whether the model and scaler loaded successfully       |
| `POST` | `/predict` | Predicts the next hourly value from 24 historical observations |

### Health Check

``` bash
curl https://energy-forecasting-mlops.onrender.com/health
```

A ready service returns a response similar to:

``` json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### Prediction Input

The `/predict` endpoint expects exactly 24 **raw consumption values in
MW**. The API applies the saved scaler internally and returns the
prediction in MW.

Example request to a local API:

``` bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      15000, 15200, 15400, 15300, 15500, 15800,
      16000, 16200, 16100, 15900, 15700, 15600,
      15500, 15400, 15600, 15800, 16100, 16400,
      16600, 16500, 16300, 16100, 15900, 15800
    ]
  }'
```

Example response from a previous successful request:

``` json
{
  "status": "success",
  "predicted_value": 19498.986328125
}
```

The prediction depends on the input sequence and the loaded model. When
running locally, Swagger UI is available at
`http://localhost:8001/docs`.

## Technology Stack

| Category         | Technologies     |
|------------------|------------------|
| Language         | Python           |
| Data processing  | pandas, NumPy    |
| Machine learning | scikit-learn     |
| Deep learning    | PyTorch          |
| API              | FastAPI, Uvicorn |
| Validation       | Pydantic         |
| Persistence      | PyTorch, joblib  |
| Testing          | pytest           |
| Containerization | Docker           |
| CI               | GitHub Actions   |
| Deployment       | Render           |

## Project Structure

``` text
energy-forecasting-mlops/
├── .github/workflows/ci.yml
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

## Installation and Local Usage

### Prerequisites

Use a Python version compatible with the pinned dependencies in
`requirements.txt`. You also need pip and Git. Docker is optional.

### 1. Clone the Repository

``` bash
git clone https://github.com/RasulAramian/energy-forecasting-mlops.git
cd energy-forecasting-mlops
```

### 2. Create and Activate a Virtual Environment

On Linux or macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the Model

Ensure `data/energy_consumption.csv` is present, then run:

``` bash
python -m src.train
```

Training saves the selected model weights and fitted scaler under
`models/`.

### 5. Evaluate the Model

``` bash
python -m src.evaluate
```

The script reports MAE and RMSE in MW on the test portion generated by
the preprocessing pipeline.

### 6. Start the API

Ensure these files exist: `models/energy_lstm_model.pth` and
`models/scaler.pkl`.

``` bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Local endpoints:

- Health check: `http://localhost:8001/health`
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Docker

The repository includes a Dockerfile for packaging the API and its
dependencies.

### Build the Image

``` bash
docker build -t energy-forecasting-mlops .
```

### Run the Container

``` bash
docker run --rm -p 8001:8001 --name energy-forecasting-api energy-forecasting-mlops
```

Open `http://localhost:8001/docs`. The model and scaler must be included
in the image or provided through a documented runtime mechanism.

## Testing

Run the test suite from the repository root:

``` bash
pytest -v
```

The tests cover API health behavior, invalid sequence lengths,
prediction behavior when artifacts are available, unavailable-model
handling, and model output shape.

## MLOps and Engineering Practices

- **Chronological splitting:** preserves time order across training,
  validation, and test sets.
- **Leakage-aware scaling:** fits the scaler on training observations
  only.
- **Validation-based model selection:** saves the model with the best
  validation loss.
- **Model persistence:** stores model weights and the fitted scaler.
- **Model serving:** exposes inference through FastAPI.
- **Input validation:** validates requests with Pydantic.
- **Automated testing:** uses pytest for API and model checks.
- **Containerization:** provides a Dockerfile.
- **Continuous integration:** includes a GitHub Actions workflow.
- **Deployment:** serves the API from Render.

## Current Limitations and Next Steps

- Reproduce earlier reported metrics before documenting them as verified
  results.
- Record the test period and dataset version in evaluation reports.
- Review duplicate timestamps and daylight-saving-time edge cases.
- Compare the LSTM against clearly implemented and re-evaluated
  baselines.
- Add plots of predictions versus actual consumption.
- Evaluate performance with rolling-window backtesting and across
  seasons.
- Improve logging and monitoring for production inference.
- Verify Docker build and runtime behavior in a clean environment.
- Track dataset, model, and experiment versions.

## Key Takeaways

This project brings together time-series forecasting and practical ML
engineering. It demonstrates preparing real-world time-series data,
training a PyTorch LSTM, evaluating predictions, serving a model through
a REST API, writing automated tests, and preparing a deployment
workflow.

The evaluation metrics above reflect the latest run available during
documentation review. Further reproducibility and validation work
remains before making broader performance claims.

## Author

**Rasul Aramian**

PhD in Mathematics

Applying mathematical and programming skills to machine learning, deep
learning, and practical ML engineering.

- **GitHub:** [RasulAramian](https://github.com/RasulAramian)
- **Project Repository:**
  [energy-forecasting-mlops](https://github.com/RasulAramian/energy-forecasting-mlops)
