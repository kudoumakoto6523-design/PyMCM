import pandas as pd
import numpy as np
from pathlib import Path


class MCMData:
    """
    Standard Data Container for PyMCM.

    This class enforces 'Tidy Data' principles:
    1. Each row is an observation.
    2. Each column is a variable.
    3. All feature data must be numeric.

    It also handles:
    - Automatic loading from CSV/Excel.
    - Automatic missing value detection and imputation.
    - Separation of Features (X) and Target (y).
    """

    def __init__(self, data_source, target=None, index_col=None, impute_strategy=None):
        """
        Initialize the data container.

        Args:
            data_source (str or pd.DataFrame): Path to .csv/.xlsx file or a pandas DataFrame.
            target (str, optional): The name of the target column (y) for prediction tasks.
            index_col (str, optional): Column name to use as the index (ID/Label).
                                       This column will be excluded from the feature matrix X.
            impute_strategy (str, optional): Strategy to handle missing values (NaN) automatically.
                                             Options: 'mean', 'median', 'zero', 'drop'.
                                             Default is None (no action, just warn).
        """
        # 1. Load Data
        if isinstance(data_source, str):
            file_path = Path(data_source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {data_source}")

            if file_path.suffix == '.csv':
                self.df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported format. Please use CSV or Excel.")
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            raise TypeError("Input must be a file path (str) or a pandas DataFrame.")

        # 2. Handle Index (Set ID column as index so it doesn't interfere with calculation)
        if index_col:
            if index_col in self.df.columns:
                self.df.set_index(index_col, inplace=True)
            else:
                raise ValueError(f"Index column '{index_col}' not found in data.")

        # 3. Handle Target Column
        self.target_name = target

        # 4. Strict Validation (Ensure all remaining columns are numeric)
        # We check the WHOLE dataframe first.
        numeric_df = self.df.select_dtypes(include=[np.number])

        # If the shape doesn't match, it means some columns were non-numeric (strings/objects)
        if numeric_df.shape[1] != self.df.shape[1]:
            dirty_cols = list(set(self.df.columns) - set(numeric_df.columns))
            raise ValueError(
                f"Data Error: Non-numeric columns detected: {dirty_cols}.\n"
                f"PyMCM only accepts numeric columns for modeling. \n"
                f"Tip: If these are IDs, pass them as `index_col`."
            )

        # 5. Handle Missing Values (Detection & Auto-Imputation)
        self.has_nan = self.df.isnull().values.any()

        if impute_strategy:
            # If user provided a strategy, fix it immediately
            self.impute(impute_strategy)
        elif self.has_nan:
            # If no strategy provided, just warn the user
            missing_count = self.df.isnull().sum().sum()
            print(f"⚠️ [WARNING] Data contains {missing_count} missing values (NaN/None)!")
            print("   -> Recommended: Pass `impute_strategy='mean'` in init, or call `data.impute(...)` manually.")

        # 6. Initial Split (X and y)
        self._refresh_split()

    def _refresh_split(self):
        """
        Internal helper to refresh X and y attributes after data changes (e.g., imputation).
        """
        if self.target_name:
            if self.target_name not in self.df.columns:
                if self.df.empty:
                    raise ValueError("Data is empty after processing (all rows dropped?).")
                raise ValueError(f"Target column '{self.target_name}' lost during processing.")

            self.y = self.df[self.target_name].values
            self.X = self.df.drop(columns=[self.target_name]).values
            self.feature_names = self.df.drop(columns=[self.target_name]).columns.tolist()
        else:
            # Unsupervised mode
            self.y = None
            self.X = self.df.values
            self.feature_names = self.df.columns.tolist()

    def impute(self, strategy='mean'):
        """
        Fix missing values in the dataset.

        Args:
            strategy (str): The imputation method.
                - 'drop': Remove rows containing NaN.
                - 'mean': Fill NaN with column mean.
                - 'median': Fill NaN with column median.
                - 'zero': Fill NaN with 0.
        """
        if not self.df.isnull().values.any():
            print("[Impute] Data is already clean. No action taken.")
            return

        original_shape = self.df.shape

        if strategy == 'drop':
            self.df.dropna(inplace=True)
            print(f"[Impute] Dropped rows with missing values. Shape: {original_shape} -> {self.df.shape}")

        elif strategy == 'mean':
            self.df.fillna(self.df.mean(), inplace=True)
            print("[Impute] Filled missing values with Mean.")

        elif strategy == 'median':
            self.df.fillna(self.df.median(), inplace=True)
            print("[Impute] Filled missing values with Median.")

        elif strategy == 'zero':
            self.df.fillna(0, inplace=True)
            print("[Impute] Filled missing values with 0.")

        else:
            raise ValueError("Strategy must be 'drop', 'mean', 'median', or 'zero'")

        # Update status and refresh split
        self.has_nan = False
        self._refresh_split()

    def get_X(self):
        """Returns the feature matrix (numpy array)."""
        return self.X

    def get_y(self):
        """Returns the target vector (numpy array)."""
        return self.y

    def summary(self):
        """Prints a summary report of the data."""
        print("--- PyMCM Data Summary ---")
        print(f"Samples: {self.X.shape[0]}, Features: {self.X.shape[1]}")
        if self.target_name:
            print(f"Target (y): {self.target_name}")

        print(f"Feature Names: {self.feature_names}")

        if self.has_nan:
            print("⚠️ STATUS: DIRTY (Contains NaN). Please impute!")
        else:
            print("✅ STATUS: CLEAN")