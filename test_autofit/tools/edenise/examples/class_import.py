class MyClass:
    @property
    def slim(self) -> Union[('AbstractGrid1D', 'Grid1D')]:
        """
        Return a `Grid1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size, 2].

        If it is already stored in its `slim` representation  the `Grid1D` is returned as it is. If not, it is
        mapped from  `native` to `slim` and returned as a new `Grid1D`.
        """
        from .non_linear.grid.grid_search import GridSearch as SearchGridSearch
        from autoconf.tools.decorators import CachedProperty