import datetime as dt
import os
from os import path
import time


class Timer:

    def __init__(self, search_internal: str):
        """Times the run-time of the non-linear searches, by outputting a start-time file to the hard-disk and using
        this to determine the total run time when a `NonLinearSearch` update is performed.

        Parameters
        ----------
        search_internal
            The directory in which the timer should save results
        """

        self.search_internal = search_internal
        os.makedirs(
            search_internal,
            exist_ok=True
        )

    def start(self):
        """
        Record the start time of a `NonLinearSearch` as universal date time, so that the run-time of the search can be
        recorded.
        """

        start_time_path = path.join(
            self.search_internal,
            ".start_time"
        )
        try:
            with open(start_time_path) as f:
                float(f.read())
        except FileNotFoundError:
            start = time.time()
            with open(start_time_path, "w+") as f:
                f.write(str(start))

    def update(self):
        """
        Update the timer of the `NonLinearSearch` so it reflections how long the `NonLinearSearch` ahs been running.
        """

        try:
            execution_time = str(time.time() - float(self.start_time))
        except TypeError:
            return

        with open(
                path.join(self.search_internal, ".time"), "w+"
        ) as f:
            f.write(execution_time)

    @property
    def start_time(self):
        """
        Load the start time written to hard disk from the .start_time file.
        """
        try:
            with open(
                    path.join(self.search_internal, ".start_time"), "r"
            ) as f:
                return f.read()
        except FileNotFoundError:
            return None

    @property
    def time(self):
        """
        Load the total time of the `NonLinearSearch` written to hard disk fom the .start_time file.
        """
        try:
            with open(
                    path.join(self.search_internal, ".time"), "r"
            ) as f:
                return f.read()
        except FileNotFoundError:
            return None
