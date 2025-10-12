"""
Data Processing Module
Handles time series data loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TimeSeriesDataProcessor:
    """
    Time series data processor with feature engineering
    """

    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.encoders = {}

    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from various formats"""
        path = Path(path)

        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Loaded data from {path}: {df.shape}")
        return df

    def generate_sample_data(self, n_series: int = 10, n_periods: int = 365) -> pd.DataFrame:
        """Generate sample time series data for testing"""

        np.random.seed(42)
        data = []

        for series_id in range(n_series):
            # Generate time series with trend and seasonality
            dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')

            # Base trend
            trend = np.linspace(100, 200, n_periods)

            # Seasonal component
            seasonal = 20 * np.sin(2 * np.pi * np.arange(n_periods) / 365)

            # Weekly pattern
            weekly = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 7)

            # Noise
            noise = np.random.normal(0, 5, n_periods)

            # Combine components
            values = trend + seasonal + weekly + noise

            # Add some series-specific characteristics
            values *= (1 + 0.1 * series_id)  # Different scales

            for i, (date, value) in enumerate(zip(dates, values)):
                data.append({
                    'date': date,
                    'series_id': f'series_{series_id:03d}',
                    'value': max(0, value),  # Ensure positive values
                    'category': f'category_{series_id % 3}',
                    'region': f'region_{series_id % 5}',
                    'promotion': np.random.binomial(1, 0.1),
                    'holiday': int(date.weekday() >= 5)  # Weekend as holiday
                })

        df = pd.DataFrame(data)
        logger.info(f"Generated sample data: {df.shape}")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        df = df.copy()

        # Convert date column
        date_col = self.config["data"]["time_column"]
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort by time and series
        df = df.sort_values([
            self.config["data"]["id_column"],
            date_col
        ]).reset_index(drop=True)

        # Handle missing values
        target_col = self.config["data"]["target_column"]

        # Forward fill missing targets within each series
        df[target_col] = df.groupby(
            self.config["data"]["id_column"]
        )[target_col].fillna(method='pad')

        # Remove remaining nulls
        initial_len = len(df)
        df = df.dropna(subset=[target_col])
        removed = initial_len - len(df)

        if removed > 0:
            logger.info(f"Removed {removed} rows with missing target values")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based and lag features"""
        df = df.copy()
        date_col = self.config["data"]["time_column"]

        # Time-based features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['hour'] = df[date_col].dt.hour
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Create time index
        df['date_idx'] = (
            df[date_col] - df[date_col].min()
        ).dt.total_seconds() / 3600  # Hours since start

        # Seasonal features
        df['quarter'] = df[date_col].dt.quarter
        df['week_of_year'] = df[date_col].dt.isocalendar().week

        # Lag features (within each series)
        target_col = self.config["data"]["target_column"]
        id_col = self.config["data"]["id_column"]

        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df.groupby(id_col)[target_col].shift(lag)

        # Rolling statistics
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df.groupby(id_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(id_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        logger.info(f"Created features. New shape: {df.shape}")
        return df

    def create_time_series_splits(
        self, 
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test splits"""

        df = df.sort_values([
            self.config["data"]["id_column"],
            self.config["data"]["time_column"]
        ])

        # Get unique time points
        unique_times = df[self.config["data"]["time_column"]].unique()
        unique_times = np.sort(unique_times)

        # Calculate split indices
        n_times = len(unique_times)
        train_end_idx = int(n_times * train_ratio)
        val_end_idx = int(n_times * (train_ratio + val_ratio))

        train_end_time = unique_times[train_end_idx - 1]
        val_end_time = unique_times[val_end_idx - 1]

        # Create splits
        train_df = df[df[self.config["data"]["time_column"]] <= train_end_time]
        val_df = df[
            (df[self.config["data"]["time_column"]] > train_end_time) &
            (df[self.config["data"]["time_column"]] <= val_end_time)
        ]
        test_df = df[df[self.config["data"]["time_column"]] > val_end_time]

        logger.info(f"Data splits: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline"""
        logger.info("Starting data processing pipeline...")

        # Clean
        df = self.clean_data(df)

        # Engineer features
        df = self.engineer_features(df)

        logger.info("Data processing completed")
        return df

    def save_processed_data(self, df: pd.DataFrame, path: str):
        """Save processed data"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if path.endswith('.parquet'):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"Saved processed data to {path}")


def main():
    """Example usage"""
    config = {
        "data": {
            "time_column": "date",
            "target_column": "value", 
            "id_column": "series_id",
            "categorical_features": ["category", "region"],
            "numerical_features": ["promotion", "holiday"]
        }
    }

    processor = TimeSeriesDataProcessor(config)

    # Generate sample data
    df = processor.generate_sample_data()

    # Process data
    processed_df = processor.process(df)

    # Create splits
    train_df, val_df, test_df = processor.create_time_series_splits(processed_df)

    # Save data
    processor.save_processed_data(train_df, "data/processed/train.csv")
    processor.save_processed_data(val_df, "data/processed/val.csv") 
    processor.save_processed_data(test_df, "data/processed/test.csv")

    print("Data processing completed!")


if __name__ == "__main__":
    main()
