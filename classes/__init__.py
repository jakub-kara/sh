import os

__all__ = []
dirname = os.path.dirname(os.path.abspath(__file__))

for f in os.listdir(dirname):
    if f != "__init__.py" and os.path.isfile(f"{dirname}/{f}") and f.endswith(".py"):
        __all__.append(f[:-3])