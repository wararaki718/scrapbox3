import joblib


class Sample:
    def __init__(self):
        self.model = joblib.load("model.pkl")
    
    def method(self, text: str) -> float:
        return self.model.method(text)
