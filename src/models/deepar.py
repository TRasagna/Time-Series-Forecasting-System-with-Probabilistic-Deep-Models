"""
DeepAR Model Implementation
Probabilistic forecasting with autoregressive recurrent networks
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import numpy as np
from typing import Dict, List, Optional, Tuple
import mlflow
import mlflow.pytorch


class DeepARModel(pl.LightningModule):
    """
    DeepAR implementation using PyTorch Lightning
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        prediction_length: int = 30,
        context_length: int = 168,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        learning_rate: float = 0.001,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.quantiles = quantiles
        self.learning_rate = learning_rate

        # Loss function for probabilistic forecasting
        self.loss_fn = QuantileLoss(quantiles=quantiles)

    def setup_model(self, training_data: TimeSeriesDataSet):
        """Setup the actual DeepAR model"""
        self.model = DeepAR.from_dataset(
            training_data,
            hidden_size=self.hidden_size,
            rnn_layers=self.num_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            loss=self.loss_fn,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        return self

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss = self.model.training_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss = self.model.validation_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers"""
        return self.model.configure_optimizers()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step"""
        return self.model.predict_step(batch, batch_idx, dataloader_idx)

    def predict(
        self, 
        dataloader, 
        return_y: bool = False,
        return_x: bool = False
    ):
        """Generate predictions"""
        return self.model.predict(
            dataloader, 
            return_y=return_y,
            return_x=return_x
        )

    def save_model(self, path: str, log_to_mlflow: bool = True):
        """Save model with MLflow tracking"""
        torch.save(self.state_dict(), path)

        if log_to_mlflow:
            mlflow.pytorch.log_model(
                self,
                "model",
                registered_model_name="DeepAR",
                signature=None
            )
            mlflow.log_artifact(path)

    @classmethod
    def load_model(cls, path: str, **kwargs):
        """Load model from checkpoint"""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model

    def get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.numel() * param.data.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.numel() * buffer.data.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb


class DeepARTrainer:
    """
    Trainer class for DeepAR model with MLflow integration
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.trainer = None

    def prepare_data(self, df) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Prepare training and validation datasets"""

        # Define the dataset configuration
        max_encoder_length = self.config["architecture"]["context_length"]
        max_prediction_length = self.config["architecture"]["prediction_length"]

        # Create training dataset
        training = TimeSeriesDataSet(
            df[lambda x: x.date <= df.date.quantile(0.8)],
            time_idx="date_idx",
            target=self.config["data"]["target_column"],
            group_ids=[self.config["data"]["id_column"]],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=self.config["data"].get("categorical_features", []),
            static_reals=self.config["data"].get("numerical_features", []),
            time_varying_known_categoricals=["day_of_week", "month"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[self.config["data"]["target_column"]],
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            df, 
            predict=True, 
            stop_randomization=True
        )

        return training, validation

    def train(
        self, 
        train_dataloader, 
        val_dataloader, 
        training_data: TimeSeriesDataSet
    ):
        """Train the model"""

        # Setup model
        self.model = DeepARModel(**self.config["architecture"])
        self.model.setup_model(training_data)

        # Setup trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config["training"]["epochs"],
            accelerator="auto",
            devices="auto",
            gradient_clip_val=self.config["training"]["grad_clip_val"],
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config["training"]["early_stopping_patience"],
                    mode="min"
                ),
                pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
                pl.callbacks.ModelCheckpoint(
                    monitor="val_loss",
                    filename="deepar-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=3,
                    mode="min"
                )
            ],
            logger=False
        )

        # Train model
        self.trainer.fit(
            self.model.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        return self.model
