
def predict_vietocr(detector, image) -> tuple:
    pred, prob = detector.predict(image , return_prob = True)
    return (pred, prob)