def prdiction(predictions):
    xgb_probability = predictions['x']
    rnn_probability = predictions['l']

    threshold = 0.5

    if rnn_probability >= threshold or xgb_probability >= threshold:
        return True
    else:
        return False