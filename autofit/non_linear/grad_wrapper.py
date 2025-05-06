from autoconf import cached_property

class GradWrapper:
    def __init__(self, function):
        self.function = function

    @cached_property
    def grad(self):
        import jax
        from jax import grad
        print("Compiling gradient")
        return jax.jit(grad(self.function))

    def __getstate__(self):
        return {"function": self.function}

    def __setstate__(self, state):
        self.__init__(state["function"])

    def __call__(self, *args, **kwargs):
        return self.grad(*args, **kwargs)