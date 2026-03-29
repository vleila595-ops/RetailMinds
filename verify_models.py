
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.model_factory import ModelFactory
    from models.arima_model import ARIMAModel
    from models.prophet_model import ProphetModel
    from models.xgboost_model import XGBoostModel

    print("Verifying model loading...")

    models_to_test = [
        ("ARIMA", ARIMAModel),
        ("Prophet", ProphetModel),
        ("XGBoost", XGBoostModel)
    ]

    all_loaded = True

    for model_name, model_class in models_to_test:
        try:
            model_instance = model_class()
            print(f"✓ {model_name} model instantiated successfully")
        except Exception as e:
            print(f"✗ {model_name} model failed to instantiate: {e}")
            all_loaded = False

    try:
        factory = ModelFactory()
        print("✓ ModelFactory instantiated successfully")

        expected_models = ['ARIMA', 'Prophet', 'XGBoost']
        for model_name in expected_models:
            if model_name in factory.models and factory.models[model_name] is not None:
                print(f"✓ {model_name} model loaded in ModelFactory")
            else:
                print(f"✗ {model_name} model not found in ModelFactory")
                all_loaded = False

    except Exception as e:
        print(f"✗ ModelFactory failed to instantiate: {e}")
        all_loaded = False

    if all_loaded:
        print("\nAll models are loaded successfully!")
        sys.exit(0)
    else:
        print("\nSome models failed to load!")
        sys.exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)