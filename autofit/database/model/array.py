import numpy as np

from autoconf.class_path import get_class_path, get_class
from .model import Base
from ..sqlalchemy_ import sa


class Array(Base):
    """
    A serialised numpy array
    """

    __tablename__ = "array"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    id = sa.Column(sa.Integer, primary_key=True)

    name = sa.Column(sa.String)

    bytes = sa.Column(sa.LargeBinary)
    _dtype = sa.Column(sa.String)
    _shape = sa.Column(sa.String)

    fit_id = sa.Column(sa.String, sa.ForeignKey("fit.id"))
    fit = sa.orm.relationship("Fit", uselist=False)

    array_type = sa.Column(sa.String)

    __mapper_args__ = {
        "polymorphic_identity": "array",
        "polymorphic_on": "array_type",
    }

    @property
    def shape(self):
        return tuple(map(int, self._shape.split(",")))

    @shape.setter
    def shape(self, shape):
        self._shape = ",".join(map(str, shape))

    @property
    def array(self):
        return np.frombuffer(
            self.bytes,
            dtype=self.dtype,
        ).reshape(self.shape)

    @array.setter
    def array(self, array):
        self._dtype = get_class_path(getattr(np, array.dtype.name))
        self.shape = array.shape
        self.bytes = array.tobytes()

    @property
    def dtype(self):
        return get_class(self._dtype)

    @property
    def value(self):
        return self.array


class HDU(Array):
    """
    A serialised astropy.io.fits.PrimaryHDU
    """

    __tablename__ = "hdu"

    id = sa.Column(sa.Integer, sa.ForeignKey("array.id"), primary_key=True)

    _header = sa.Column(sa.String)

    __mapper_args__ = {
        "polymorphic_identity": "hdu",
    }

    @property
    def header(self):
        """
        The header of the HDU
        """
        from astropy.io.fits import Header

        return Header.fromstring(self._header)

    @header.setter
    def header(self, header):
        self._header = header.tostring()

    @property
    def hdu(self):
        from astropy.io import fits

        return fits.PrimaryHDU(
            self.array,
            self.header,
        )

    @hdu.setter
    def hdu(self, hdu):
        self.array = hdu.data
        self.header = hdu.header

    @property
    def value(self):
        return self.hdu

    @property
    def dtype(self):
        return np.dtype(">f8")
