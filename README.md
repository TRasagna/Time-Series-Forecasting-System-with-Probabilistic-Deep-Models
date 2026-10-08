# ProbForecast — Scalable Probabilistic Time-Series Forecasting System

[![CI/CD](https://github.com/TRasagna/Time-Series-Forecasting-System-with-Probabilistic-Deep-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/TRasagna/Time-Series-Forecasting-System-with-Probabilistic-Deep-Models/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready deep learning-based forecasting platform using probabilistic models (DeepAR, Temporal Fusion Transformer, N-BEATS) with MLOps best practices.

## 🚀 Quick Start

```bash
# Clone and setup
unzip probforecast-complete.zip
cd probforecast
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate sample data
python src/data/dataset.py

# Train a model (optional - API works with mock data by default)
python src/train/train_loop.py --config configs/training.yaml

# Start services
docker-compose up --build

# OR run individual services:
# API: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# Dashboard: streamlit run src/dashboard/app.py
```

## 🌐 Service URLs
- **API**: http://localhost:8000 (docs: http://localhost:8000/docs)
- **Dashboard**: http://localhost:8501  
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

## 📊 Features

### Core ML Capabilities
- ✅ **DeepAR**: Probabilistic RNN-based forecasting
- ✅ **Temporal Fusion Transformer (TFT)**: Multi-horizon interpretable forecasting  
- ✅ **N-BEATS**: Pure deep learning with trend/seasonality decomposition
- ✅ **Uncertainty Quantification**: Confidence intervals and quantile forecasts
- ✅ **Multi-variate Support**: Thousands of correlated time series
- ✅ **Hyperparameter Optimization**: Automated tuning with Optuna

### Production MLOps
- ✅ **Experiment Tracking**: MLflow integration
- ✅ **Data Versioning**: DVC pipeline support
- ✅ **Model Registry**: Automated model promotion
- ✅ **CI/CD**: GitHub Actions workflows
- ✅ **Containerization**: Docker + Kubernetes manifests
- ✅ **Monitoring**: Drift detection with EvidentlyAI
- ✅ **Observability**: Prometheus metrics + Grafana dashboards

### API & UI
- ✅ **REST API**: FastAPI with automatic docs
- ✅ **Interactive Dashboard**: Streamlit visualization
- ✅ **Caching**: Redis for performance
- ✅ **Health Checks**: Kubernetes-ready probes

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Pipeline   │───▶│   ML Training   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│  FastAPI Service │◀───│ Model Registry  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
┌─────────────────┐    ┌──────────────────┐
│ Grafana + Prom  │    │ Streamlit UI     │
└─────────────────┘    └──────────────────┘
```

## 📁 Project Structure

```
probforecast/
├── 📊 data/                    # Data storage
│   ├── raw/                   # Raw datasets  
│   └── processed/             # Processed datasets
├── ⚙️ configs/                 # Configuration files
│   ├── deepar.yaml           # DeepAR config
│   ├── tft.yaml              # TFT config
│   ├── nbeats.yaml           # N-BEATS config
│   └── training.yaml         # Training config
├── 🧠 src/                     # Source code
│   ├── data/                 # Data processing  
│   ├── models/               # Model implementations
│   ├── train/                # Training scripts
│   ├── api/                  # FastAPI service
│   ├── dashboard/            # Streamlit app
│   ├── monitoring/           # Drift detection
│   └── utils/                # Utilities
├── 🐳 infra/                   # Infrastructure
│   ├── docker/               # Dockerfiles
│   └── k8s/                  # Kubernetes manifests
├── 🧪 tests/                   # Test suite
├── 📓 notebooks/               # Jupyter notebooks
├── docker-compose.yml         # Local development
└── requirements.txt           # Dependencies
```

## 🎯 API Usage

### Generate Forecast
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "store_001",
    "horizon": 30,
    "model_type": "deepar",
    "quantiles": [0.1, 0.5, 0.9]
  }'
```

### Response Format
```json
{
  "series_id": "store_001",
  "forecast": [
    {
      "timestamp": "2025-10-13T00:00:00Z",
      "mean": 125.6,
      "quantiles": {
        "p10": 115.2,
        "p50": 125.6, 
        "p90": 138.9
      }
    }
  ],
  "model_version": "latest",
  "model_type": "deepar",
  "forecast_horizon": 30,
  "generated_at": "2025-10-12T17:00:00Z"
}
```

## 🚀 Deployment

### Local Development
```bash
docker-compose up --build
```

### Kubernetes
```bash
kubectl apply -f infra/k8s/
kubectl port-forward svc/probforecast-api-service 8000:80
kubectl port-forward svc/probforecast-dashboard-service 8501:80
```

### Production Scaling
- Horizontal pod autoscaling configured
- Redis cluster for caching
- PostgreSQL for metadata
- S3/MinIO for artifacts

## 📈 Model Training

### Train DeepAR
```bash
python src/train/train_loop.py --config configs/deepar.yaml
```

### Hyperparameter Optimization
```bash
python src/train/train_loop.py --config configs/deepar.yaml --optimize
```

### Custom Data
```python
from src.data.dataset import TimeSeriesDataProcessor

# Your CSV should have columns: date, series_id, value, [optional features]
config = {...}  # See configs/training.yaml
processor = TimeSeriesDataProcessor(config)

df = pd.read_csv("your_data.csv")
processed = processor.process(df)
train, val, test = processor.create_splits(processed)

# Save for training
processor.save_csv(train, "data/processed/train.csv")
```

## 📊 Monitoring & Alerts

### Drift Detection
- Data distribution changes
- Model performance degradation  
- Automated retraining triggers

### Metrics Dashboard
- Request latency and throughput
- Model accuracy over time
- Data quality scores
- Infrastructure health

### Alerting
- Slack/email notifications
- Prometheus alert rules
- Custom thresholds

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suites
pytest tests/test_api.py -v
pytest tests/test_models.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) for model implementations
- [MLflow](https://mlflow.org/) for experiment tracking
- [Evidently AI](https://evidentlyai.com/) for monitoring

---

**Made with ❤️ for the ML community**
