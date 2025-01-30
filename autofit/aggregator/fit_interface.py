from abc import ABC, abstractmethod
from typing import Any, List, Optional


class Fit(ABC):
    """
    Common interface for objects representing the result of a "fit" or "search."
    This interface is intentionally minimal, representing the smallest
    set of attributes and methods that are conceptually shared by both
    a database-backed Fit and a file-backed SearchOutput.
    """

    children: List["Fit"]

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A unique identifier for the fit/search output.
        """

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """
        The name of the fit/search output.
        """

    @property
    @abstractmethod
    def unique_tag(self) -> Optional[str]:
        """
        A unique tag for differentiating this fit/search output from others.
        """

    @property
    @abstractmethod
    def path_prefix(self) -> Optional[str]:
        """
        A path prefix or directory path used by the search or fit.
        """

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        """
        True if the fit/search completed successfully, False otherwise.
        """

    @property
    @abstractmethod
    def model(self) -> Any:
        """
        The model that was fit (database-backed or loaded from disk).
        """

    @property
    @abstractmethod
    def instance(self) -> Any:
        """
        The maximum-likelihood instance (database-backed or loaded from disk).
        """

    @property
    @abstractmethod
    def samples(self) -> Any:
        """
        The samples resulting from the fit/search.
        """

    @property
    @abstractmethod
    def latent_samples(self) -> Any:
        """
        An alternative set of samples (latent samples) if they exist, else None.
        """

    @abstractmethod
    def child_values(self, name: str) -> List[Any]:
        """
        Get the values of a given key for all children, if the concept of children applies.
        """

    @property
    @abstractmethod
    def children(self) -> List[Any]:
        """
        Return a list of child fits/searches, if the concept of children applies.
        """

    @abstractmethod
    def value(self, name: str) -> Any:
        """
        Retrieve a stored object by its name (Pickle, JSON, array, etc.).
        Returns None if no value with that name is found.
        """
