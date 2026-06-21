from abc import ABC, abstractmethod

import pandas as pd


class BaseDataPipeline(ABC):
    """Abstract base class for data fetching and processing pipelines."""

    def __init__(self):
        self._data: pd.DataFrame | None = None

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Download raw data from the source and return a DataFrame."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform the raw DataFrame."""

    def run(self) -> pd.DataFrame:
        """Convenience: fetch then clean in one call."""
        raw = self.fetch()
        self._data = self.clean(raw)
        return self._data

    def get_data(self) -> pd.DataFrame | None:
        """Return the cleaned DataFrame (None if run() hasn't been called)."""
        return self._data

    def to_csv(self, path: str, **kwargs) -> None:
        """Save the cleaned data to CSV."""
        if self._data is None:
            raise RuntimeError("No data available. Call run() first.")
        self._data.to_csv(path, index=False, **kwargs)
