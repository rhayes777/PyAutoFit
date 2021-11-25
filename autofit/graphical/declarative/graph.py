from typing import List, cast, Optional

from autofit.graphical.declarative.factor.prior import PriorFactor
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.variable import Variable
from autofit.text.formatter import TextFormatter


class DeclarativeGraphOutput:
    def __init__(
            self,
            graph: "DeclarativeFactorGraph",
            variable_formatter=str
    ):
        self.graph = graph
        self.variable_formatter = variable_formatter

    @property
    def info(self) -> str:
        """
        Describes the graph. Output in graph.info
        """
        prior_factor_info = "\n".join(
            map(
                self.info_for_prior_factor,
                self.graph.prior_factors
            )
        )
        analysis_factor_info = "\n\n".join(
            map(
                self.info_for_analysis_factor,
                self.graph.analysis_factors
            )
        )
        return f"PriorFactors\n\n{prior_factor_info}\n\nAnalysisFactors\n\n{analysis_factor_info}"

    def _related_factor_names(
            self,
            variable: Variable,
            excluded_factor: Optional[Factor] = None
    ) -> str:
        """

        Parameters
        ----------
        variable
            A variable in the graph
        excluded_factor
            A factor which should not be included. e.g. the factor
            for which the variable is being checked.

        Returns
        -------
        A string describing the other factor's relationship to the variable.
        """
        related_factors = self.graph.related_factors(
            variable,
            excluded_factor=excluded_factor
        )

        return ", ".join(
            factor.name_for_variable(variable)
            for factor in related_factors
        )

    def info_for_prior_factor(
            self,
            prior_factor: PriorFactor
    ) -> str:
        """
        A string describing a given PriorFactor in the context of this graph.
        """
        related_factor_names = self._related_factor_names(
            variable=prior_factor.variable,
            excluded_factor=prior_factor
        )

        formatter = TextFormatter()
        formatter.add(
            (f"{prior_factor.name} ({related_factor_names})",),
            prior_factor.variable
        )
        return formatter.text

    def info_for_analysis_factor(
            self,
            analysis_factor
    ) -> str:
        """
        A string describing a given AnalysisFactor in the context of this graph.
        """
        model = analysis_factor.prior_model
        formatter = TextFormatter()

        for path, prior in model.path_instance_tuples_for_class(
                Prior,
                ignore_children=True
        ):
            name = path[-1]
            related_factor_names = self._related_factor_names(
                prior,
                excluded_factor=analysis_factor
            )
            path = path[:-1] + (f"{name} ({related_factor_names})",)
            formatter.add(path, prior)

        return f"{analysis_factor.name}\n\n{formatter.text}"


class DeclarativeFactorGraph(FactorGraph):
    @property
    def analysis_factors(self):
        """
        Analysis factors associated with this graph.
        """
        from .factor.analysis import AnalysisFactor
        return cast(
            List[AnalysisFactor],
            self._factors_with_type(
                AnalysisFactor
            )
        )

    @property
    def prior_factors(self) -> List[PriorFactor]:
        """
        Prior factors associated with this graph.
        """
        return cast(
            List[PriorFactor],
            self._factors_with_type(
                PriorFactor
            )
        )

    @property
    def info(self) -> str:
        """
        Describes the graph. Output in graph.info
        """
        return DeclarativeGraphOutput(self).info
