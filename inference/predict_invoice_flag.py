import joblib
import pandas as pd

MODEL_PATH = r"D:\SEVEN MENTOR\Data Science Projects\Vendor Invoice Intelligence System\invoice_flagging\models\predict_flag_invoice.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
# Test data (replace with your actual feature names)
    sample_invoice = {
        "invoice_quantity": [50],
        "invoice_dollars": [1200.50],
        "Freight": [45.00],
        "total_item_quantity": [50],
        "total_item_dollars": [1200.50]
    }
    
    # Run prediction
    result = predict_invoice_flag(sample_invoice)
    print(result)