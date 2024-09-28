def prdiction(predictions):
    rf_prediction = predictions['r']
    xgb_probability = predictions['x']

    threshold = 0.5
    
    if xgb_probability > threshold:
        return True
    elif rf_prediction == 1:
        return True
    else:
        return False