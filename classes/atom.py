import numpy as np

class Atom:
    def __init__(self, name: str = "", mass: float = 0):
        self._name = name
        self._mass = mass
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._acc = np.zeros(3)

    @property
    def name(self):
        return self._name

    @property
    def mass(self):
        return self._mass

    @property
    def pos_d(self): 
        return self._pos
    
    @pos_d.setter
    def pos_d(self, value: np.ndarray):
        self._pos[:] = value
    
    @property
    def vel_d(self): 
        return self._vel
    
    @vel_d.setter
    def vel_d(self, value: np.ndarray):
        self._vel[:] = value
    
    @property
    def acc_d(self):
        return self._acc
    
    @acc_d.setter
    def acc_d(self, value: np.ndarray):
        self._acc[:] = value