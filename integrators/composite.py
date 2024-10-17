from integrators.integrators import Integrator2

class CompositeIntegrator(Integrator2):
    def __init__(self):
        self._state = 0
        self._methods: dict[int, Integrator2] = {}
        self._initcount = 0

    def bind_integrator(self, method, key = None):
        if key is None:
            key = max(self._methods.keys()) + 1
        self._methods[key] = method
        return self

    def remove_integrator(self, key: int):
        self._methods.pop(key, None)
        return self

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if value in self._methods.keys():
            self._state = value

    @property
    def active(self):
        return self._methods[self._state]

    @property
    def isub(self):
        return self.active.isub

    @property
    def steps(self):
        return max([i.steps for i in self._methods.values()])

    @property
    def substeps(self):
        return self.active.substeps

    def integrate(self, *args):
        if self.state > 0:
            self._initcount += 1
        return self._methods[self._state].integrate(*args)

    def to_init(self):
        self.state = 0
        if self.active.steps > 1:
            self.state = 1
            self._initcount = 0

    def to_normal(self):
        self.state = 0
        if self._initcount < self.active.steps and self.active.steps > 1:
            self.state = 1