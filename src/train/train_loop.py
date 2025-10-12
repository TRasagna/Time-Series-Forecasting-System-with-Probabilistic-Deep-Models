"""
Training Loop Implementation
Unified training interface for all models
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import mlflow
import mlflow.pytorch
from loguru import logger

from src.models.deepar import DeepARModel, DeepARTrainer
from src.data.dataset import TimeSeriesDataProcessor


class ModelFactory:
    """Factory class for creating different model types"""

    @staticmethod
    def create_trainer(model_type: str, config: Dict) -> Any:
        """Create trainer based on model type"""
        trainers = {
            "deepar": DeepARTrainer,
        }

        if model_type not in trainers:
            raise ValueError(f"Unknown model type: {model_type}")

        return trainers[model_type](config)

    @staticmethod
    def create_model(model_type: str, config: Dict) -> pl.LightningModule:
        """Create model based on type"""
        models = {
            "deepar": DeepARModel,
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        return models[model_type](**config["architecture"])


class TrainingPipeline:
    """Main training pipeline"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_type = self.config["model"]["type"]
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories"""
        Path(self.config["model"]["save_path"]).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)

    def load_and_prepare_data(self) -> tuple:
        """Load and prepare training data"""
        logger.info("Loading and preparing data...")

        # Load data processor
        processor = TimeSeriesDataProcessor(self.config)

        # Load datasets
        train_df = processor.load_data(self.config["data"]["train_path"])
        val_df = processor.load_data(self.config["data"]["val_path"]) if self.config["data"].get("val_path") else None

        # Combine for processing if validation set is separate
        if val_df is not None:
            full_df = pd.concat([train_df, val_df])
        else:
            full_df = train_df

        # Process data
        processed_df = processor.process(full_df)

        # Create trainer
        trainer = ModelFactory.create_trainer(self.model_type, self.config)

        # Prepare datasets
        training_data, validation_data = trainer.prepare_data(processed_df)

        # Create data loaders
        train_dataloader = training_data.to_dataloader(
            train=True, 
            batch_size=self.config["training"]["batch_size"], 
            num_workers=0
        )
        val_dataloader = validation_data.to_dataloader(
            train=False, 
            batch_size=self.config["training"]["batch_size"], 
            num_workers=0
        )

        return train_dataloader, val_dataloader, training_data, trainer

    def train_model(self):
        """Train the model"""
        logger.info(f"Starting training for {self.model_type} model...")

        # Load data
        train_dataloader, val_dataloader, training_data, trainer = self.load_and_prepare_data()

        # Train model
        logger.info("Training final model...")
        model = trainer.train(train_dataloader, val_dataloader, training_data)

        # Save model
        model_path = os.path.join(
            self.config["model"]["save_path"],
            f"{self.model_type}_final.pkl"
        )
        model.save_model(model_path)

        logger.success(f"Training completed! Model saved to {model_path}")

        return model

    def run(self):
        """Run the complete training pipeline"""
        logger.info("Starting training pipeline...")

        try:
            # Train model
            model = self.train_model()

            logger.success("Training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train time series forecasting model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepar", "tft", "nbeats"],
        help="Model type (overrides config)"
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = TrainingPipeline(args.config)
    if args.model:
        pipeline.config["model"]["type"] = args.model

    pipeline.run()


if __name__ == "__main__":
    main()
