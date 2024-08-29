from gym import spaces
import numpy as np

class Strings(spaces.Space):
    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values
        
    def sample(self):
        return self.np_random.choice(self._values)

    def contains(self, value):
        return value in self._dict
    
    def map(self, action):
        return self._dict[action]
    
    def __iter__(self):
        return self._values.__iter__()