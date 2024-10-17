class A:
    def __init__(self):
        self.a = "a"
        self.fun()

    def fun(self):
        print("in A")

class B(A):
    def __init__(self):
        self.b = "b"
        super().__init__()

    def fun(self):
        print("in B")

b = B()
breakpoint()