class LinearRelationship:
    def __init__(self, m: float, c: float):
        """
        Describes a linear relationship between x and y, y = mx + c

        Parameters
        ----------
        m
            The gradient of the relationship
        c
            The y-intercept of the relationship
        """
        self.m = m
        self.c = c

    def __call__(self, x: float) -> float:
        """
        Calculate the value of y for a given value of x
        """
        return self.m * x + self.c

    def __str__(self):
        return f"y = {self.m}x + {self.c}"

    def __repr__(self):
        return str(self)
