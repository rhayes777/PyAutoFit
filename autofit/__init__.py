from autoconf import conf
from autofit.mapper.model import path_instances_of_class
from autofit.mapper.prior_model.attribute_pair import (
    cast_collection,
    AttributeNameValue,
    PriorNameValue,
    InstanceNameValue,
)

dir(conf)
from . import exc
from autofit.optimize.non_linear.samples import AbstractSamples, MCMCSamples, NestedSamplerSamples
from .aggregator import Aggregator, PhaseOutput
from .mapper import *
from .mapper import link
from .mapper.model import AbstractModel
from .mapper.model import ModelInstance
from .mapper.model import ModelInstance as Instance
from .mapper.model_mapper import ModelMapper
from .mapper.model_mapper import ModelMapper as Mapper
from .mapper.model_object import ModelObject
from .mapper.prior_model import *
from .mapper.prior_model.abstract import AbstractPriorModel
from .mapper.prior_model.annotation import AnnotationPriorModel
from .mapper.prior_model.collection import CollectionPriorModel
from .mapper.prior_model.collection import CollectionPriorModel as Collection
from autofit.mapper.prior.deferred import DeferredArgument
from autofit.mapper.prior.deferred import DeferredInstance
from .mapper.prior_model.dimension_type import DimensionType, map_types
from .mapper.prior_model.prior_model import PriorModel
from .mapper.prior_model.prior_model import PriorModel as Model
from .mapper.prior_model.util import PriorModelNameValue
from .optimize.grid_search import GridSearch as OptimizerGridSearch
from .optimize import *
from .optimize.non_linear.downhill_simplex import DownhillSimplex
from .optimize.non_linear.nested_sampling.dynesty import DynestyStatic, DynestyDynamic
from .optimize.grid_search import GridSearchResult
from .optimize.non_linear.nested_sampling.multi_nest import MultiNest
from .optimize.non_linear.mock_nlo import MockNLO
from .optimize.non_linear.non_linear import Analysis
from .optimize.non_linear.non_linear import NonLinearOptimizer
from .optimize.non_linear.emcee import Emcee
from autofit.optimize.non_linear.paths import Paths
from autofit.optimize.non_linear.paths import make_path
from autofit.optimize.non_linear.paths import convert_paths
from .optimize.non_linear.non_linear import Result
from .text import formatter, samples_text
from .tools import *
from .tools import path_util
from .tools.phase import AbstractPhase
from .tools.phase import Phase
from .tools.phase import Dataset
from .tools.phase import as_grid_search
from .tools.phase_property import PhaseProperty
from .tools.pipeline import Pipeline
from .tools.pipeline import ResultsCollection
from autofit.mapper.prior import AbstractPromise
from autofit.mapper.prior import last
from .mapper.prior import *

__version__ = '0.58.0'
