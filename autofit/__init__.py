from autofit.mapper.model import path_instances_of_class
from autofit.mapper.prior_model.prior import instanceNameValue
from autofit.mapper.prior_model.prior import (
    Prior,
    UniformPrior,
    GaussianPrior,
    LogUniformPrior,
    AttributeNameValue,
)
from autofit.mapper.prior_model.prior import PriorNameValue
from autofit.mapper.prior_model.prior import cast_collection
from . import conf
from . import exc
from .aggregator import Aggregator
from .mapper import *
from .mapper import link
from .mapper.model import AbstractModel
from .mapper.model import ModelInstance
from .mapper.model_mapper import ModelMapper
from .mapper.model_object import ModelObject
from .mapper.prior_model import *
from .mapper.prior_model.abstract import AbstractPriorModel
from .mapper.prior_model.annotation import AnnotationPriorModel
from .mapper.prior_model.collection import CollectionPriorModel
from .mapper.prior_model.deferred import DeferredArgument
from .mapper.prior_model.deferred import DeferredInstance
from .mapper.prior_model.dimension_type import DimensionType, map_types
from .mapper.prior_model.prior_model import PriorModel
from .mapper.prior_model.util import PriorModelNameValue
from .optimize import *
from .optimize.non_linear.downhill_simplex import DownhillSimplex
from .optimize.non_linear.downhill_simplex import DownhillSimplex
from .optimize.non_linear.grid_search import GridSearch
from .optimize.non_linear.multi_nest import MultiNest
from .optimize.non_linear.non_linear import Analysis
from .optimize.non_linear.non_linear import NonLinearOptimizer
from .optimize.non_linear.emcee import Emcee
from .optimize.non_linear.nuts import NUTS
from autofit.optimize.non_linear.paths import Paths
from autofit.optimize.non_linear.paths import make_path
from autofit.optimize.non_linear.paths import convert_paths
from .optimize.non_linear.non_linear import Result
from .tools import *
from .tools import path_util, text_util
from .tools.phase import AbstractPhase
from .tools.phase import Phase
from .tools.phase import as_grid_search
from .tools.phase_property import PhaseProperty
from .tools.pipeline import Pipeline
from .tools.pipeline import ResultsCollection
from .tools.promise import Promise
from .tools.promise import PromiseResult
from .tools.promise import last

__version__ = '0.47.0'
