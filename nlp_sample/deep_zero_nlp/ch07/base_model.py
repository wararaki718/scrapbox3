import os
import pickle
from typing import Any, Optional

import numpy as np


class BaseModel:
    def __init__(self) -> None:
        self.params = None
        self.grads = None
    
    def forward(self, *args: Any):
        raise NotImplementedError

    def backward(self, *args: Any):
        raise NotImplementedError

    def save_params(self, filename: Optional[str]=None) -> None:
        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"
        
        params = [param.astype(np.float16) for param in self.params]

        with open(filename, 'wb') as f:
            pickle.dump(params, f)
    
    def load_params(self, filename: Optional[str]=None) -> None:
        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"
        
        if '/' in filename:
            filename = filename.replace('/', os.sep)
        
        if not os.path.exists(filename):
            raise IOError(f"No file: {filename}")
        
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        
        params = [param.astype(float) for param in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
