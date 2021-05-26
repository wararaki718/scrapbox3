class Sample:
    def __init__(self):
        pass
    
    def _load_param(self) -> int:
        return 1
    
    def test(self) -> bool:
        param = self._load_param()
        return param == 1
