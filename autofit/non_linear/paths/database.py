from .abstract import AbstractPaths


class DatabasePaths(AbstractPaths):
    def __init__(
            self,
            session,
            name="",
            path_prefix=""
    ):
        super().__init__(
            name=name,
            path_prefix=path_prefix,
        )
        self.session = session

    def save_object(self, name: str, obj: object):
        pass

    def load_object(self, name: str):
        pass

    def remove_object(self, name: str):
        pass

    def is_object(self, name: str) -> bool:
        pass

    @property
    def is_complete(self) -> bool:
        pass

    def completed(self):
        pass

    def load_samples(self):
        pass

    def load_samples_info(self):
        pass

    def save_summary(self, samples, log_likelihood_function_time):
        pass

    def save_all(self, info, pickle_files):
        pass
