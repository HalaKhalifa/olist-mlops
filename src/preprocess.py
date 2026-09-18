"""Preprocessing module for Olist MLOps service.

Applies the pre-fitted scikit-learn ColumnTransformer to transform engineered
features into the exact numerical representation required by the model.
Fitted transformers are loaded and never re-fitted during inference.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from config.logging_config import logger
from config.settings import settings
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features
from src.utils import load_artifact


class PreprocessingService:
    """Manages feature transformation using frozen pre-fitted ColumnTransformer."""

    def __init__(self, preprocessor_path: Optional[Path] = None):
        self.preprocessor_path = preprocessor_path or settings.model.preprocessor_path
        self._preprocessor: Optional[ColumnTransformer] = None
        self._feature_names: Optional[List[str]] = None

    @property
    def preprocessor(self) -> ColumnTransformer:
        """Lazy loader for ColumnTransformer artifact."""
        if self._preprocessor is None:
            logger.info(f"Loading preprocessor from {self.preprocessor_path}")
            self._preprocessor = load_artifact(self.preprocessor_path)
        return self._preprocessor

    @property
    def feature_names(self) -> List[str]:
        """Names of the 145 output features from the ColumnTransformer."""
        if self._feature_names is None:
            feature_names_file = settings.model.feature_names_path
            if feature_names_file.exists():
                with open(feature_names_file, "r") as f:
                    self._feature_names = [line.strip() for line in f if line.strip()]
            else:
                ohe = self.preprocessor.named_transformers_["categorical"]["encoder"]
                ohe_cols = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
                self._feature_names = NUMERIC_COLS + ohe_cols
        return self._feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply feature engineering and pre-fitted ColumnTransformer.

        Args:
            df: Raw orders DataFrame.

        Returns:
            Transformed feature matrix of shape (n_samples, 145).
        """
        # Step 1: Feature engineering
        engineered_df = engineer_features(df)

        # Step 2: Ensure all required columns are present
        for col in NUMERIC_COLS:
            if col not in engineered_df.columns:
                engineered_df[col] = np.nan
        for col in CATEGORICAL_COLS:
            if col not in engineered_df.columns:
                engineered_df[col] = "missing"

        # Step 3: Transform with frozen ColumnTransformer
        transformed = self.preprocessor.transform(engineered_df)
        return transformed


# Singleton preprocessor service instance
preprocessor_service = PreprocessingService()
