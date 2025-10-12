#!/bin/bash
set -e

echo "🚀 Starting ProbForecast Setup..."

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed models logs reports results

# Generate sample data
if [ ! -f "data/raw/sample_data.csv" ]; then
    echo "📊 Generating sample data..."
    python src/data/dataset.py
fi

# Start services
echo "🐳 Starting services with Docker Compose..."
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

echo ""
echo "🎉 ProbForecast is now running!"
echo ""
echo "📋 Service URLs:"
echo "   🔗 API:       http://localhost:8000"
echo "   🔗 API Docs:  http://localhost:8000/docs"
echo "   🔗 Dashboard: http://localhost:8501"
echo "   🔗 MLflow:    http://localhost:5000"
echo "   🔗 Grafana:   http://localhost:3000 (admin/admin123)"
echo ""
echo "🛑 To stop all services: docker-compose down"
