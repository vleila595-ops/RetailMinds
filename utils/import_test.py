# Import test script to check for unresolved imports
try:
    import pandas as pd
    print("[OK] pandas imported successfully")
except ImportError as e:
    print(f"[FAIL] pandas import failed: {e}")

try:
    import numpy as np
    print("[OK] numpy imported successfully")
except ImportError as e:
    print(f"[FAIL] numpy import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("[OK] matplotlib imported successfully")
except ImportError as e:
    print(f"[FAIL] matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("[OK] seaborn imported successfully")
except ImportError as e:
    print(f"[FAIL] seaborn import failed: {e}")

try:
    from datetime import datetime
    print("[OK] datetime imported successfully")
except ImportError as e:
    print(f"[FAIL] datetime import failed: {e}")

try:
    import warnings
    print("[OK] warnings imported successfully")
except ImportError as e:
    print(f"[FAIL] warnings import failed: {e}")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    print("[OK] SARIMAX imported successfully")
except ImportError as e:
    print(f"[FAIL] SARIMAX import failed: {e}")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    print("[OK] seasonal_decompose imported successfully")
except ImportError as e:
    print(f"[FAIL] seasonal_decompose import failed: {e}")

try:
    from statsmodels.tsa.stattools import adfuller
    print("[OK] adfuller imported successfully")
except ImportError as e:
    print(f"[FAIL] adfuller import failed: {e}")

try:
    from prophet import Prophet
    print("[OK] Prophet imported successfully")
except ImportError as e:
    print(f"[FAIL] Prophet import failed: {e}")

try:
    from xgboost import XGBRegressor
    print("[OK] XGBRegressor imported successfully")
except ImportError as e:
    print(f"[FAIL] XGBRegressor import failed: {e}")

try:
    from sklearn.preprocessing import StandardScaler
    print("[OK] StandardScaler imported successfully")
except ImportError as e:
    print(f"[FAIL] StandardScaler import failed: {e}")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    print("[OK] sklearn metrics imported successfully")
except ImportError as e:
    print(f"[FAIL] sklearn metrics import failed: {e}")

try:
    import pmdarima as pm
    print("[OK] pmdarima imported successfully")
except ImportError as e:
    print(f"[FAIL] pmdarima import failed: {e}")

try:
    import plotly
    print("[OK] plotly imported successfully")
except ImportError as e:
    print(f"[FAIL] plotly import failed: {e}")

print("\nImport test completed.")