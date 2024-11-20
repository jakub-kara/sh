class ArrayShapeError(ValueError):
    def __init__(self, allowed, entered):
        super().__init__(f"Array of shape {allowed} expected, but {entered} was received.")

    @staticmethod
    def check_shape(allowed, entered):
        if allowed != entered: raise ArrayShapeError(allowed, entered)