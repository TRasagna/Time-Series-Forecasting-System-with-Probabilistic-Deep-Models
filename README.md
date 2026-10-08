# ProbForecast: Probabilistic Time-Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A probabilistic forecasting system built around **DeepAR** (PyTorch Forecasting + PyTorch Lightning), with experiment tracking in MLflow, a FastAPI forecasting service, and a Streamlit dashboard.

## What's implemented

| Component | File | Details |
|---|---|---|
| Data pipeline | `src/data/dataset.py` | `TimeSeriesDataProcessor`: loads CSVs or generates synthetic multi-series data, cleans it, engineers time features, and builds time-based train/val/test splits |
| DeepAR model | `src/models/deepar.py` | Lightning wrapper around PyTorch Forecasting's `DeepAR` with quantile outputs; save/load and MLflow model logging |
| Training pipeline | `src/train/train_loop.py` | Config-driven training (`configs/training.yaml`) with MLflow experiment tracking |
| Forecast API | `src/api/app.py` | FastAPI service with `/forecast`, `/health`, and Prometheus `/metrics`; Redis caching of forecasts |
| Dashboard | `src/dashboard/app.py` | Streamlit + Plotly UI for requesting forecasts and viewing prediction intervals |
| Infrastructure | `docker-compose.yml`, `infra/docker/` | API, dashboard, MLflow, Postgres, Redis, Prometheus, and Grafana services |

## Current status and roadmap
- ✅ DeepAR training pipeline with MLflow tracking
- ✅ API, dashboard, caching, and metrics endpoints
- ⏳ The `/forecast` endpoint currently returns **simulated probabilistic forecasts** for demo purposes; next step is serving the trained DeepAR checkpoint from MLflow
- ⏳ Configs for **Temporal Fusion Transformer** and **N-BEATS** are included (`configs/tft.yaml`, `configs/nbeats.yaml`); model implementations are planned
- ⏳ Evaluation (CRPS, quantile loss, coverage) and automated tests

## Project structure
```
configs/            model and training configs (deepar, tft, nbeats, training)
src/data/           data loading, cleaning, feature engineering
src/models/         DeepAR model
src/train/          training pipeline
src/api/            FastAPI forecasting service
src/dashboard/      Streamlit dashboard
infra/docker/       Dockerfiles and Prometheus config
notebooks/          quick-start notebook
```

## Getting started
```bash
git clone https://github.com/TRasagna/Time-Series-Forecasting-System-with-Probabilistic-Deep-Models.git
cd Time-Series-Forecasting-System-with-Probabilistic-Deep-Models
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate sample data
python src/data/dataset.py

# Train DeepAR
python src/train/train_loop.py --config configs/training.yaml

# Run the API and dashboard
uvicorn src.api.app:app --port 8000
streamlit run src/dashboard/app.py

# Or run the full stack
docker-compose up --build
```

## Tech stack
Python · PyTorch · PyTorch Lightning · PyTorch Forecasting · MLflow · FastAPI · Streamlit · Plotly · Redis · Docker · Prometheus · Grafana

## License
MIT. See [LICENSE](LICENSE).
