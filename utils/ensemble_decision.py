def prdiction(predictions):
    rf_prediction = predictions['r']
    xgb_probability = predictions['x']

    threshold = 0.4

    if rf_prediction == 1 and xgb_probability >= threshold:
        return True 
    elif xgb_probability >= threshold:
        return True
    elif rf_prediction == 0 and xgb_probability < threshold:
        return False  
    else:
        return False