from typing import List, cast

from autofit.graphical.declarative.factor.prior import PriorFactor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.mapper.prior.abstract import Prior
from autofit.text.formatter import TextFormatter


class DeclarativeFactorGraph(FactorGraph):
    @property
    def analysis_factors(self):
        from .factor.analysis import AnalysisFactor
        return cast(
            List[AnalysisFactor],
            self._factors_with_type(
                AnalysisFactor
            )
        )

    @property
    def prior_factors(self) -> List[PriorFactor]:
        return cast(
            List[PriorFactor],
            self._factors_with_type(
                PriorFactor
            )
        )

    def _related_factor_names(
            self,
            variable,
            excluded_factor=None
    ):
        related_factors = self.related_factors(
            variable,
            excluded_factor=excluded_factor
        )

        return ", ".join(
            factor.name_for_variable(variable)
            for factor in related_factors
        )

    def _info_for_prior_factor(
            self,
            prior_factor: PriorFactor
    ):
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

    def _info_for_analysis_factor(
            self,
            analysis_factor
    ):
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

    @property
    def info(self) -> str:
        """
        Describes the graph. Output in graph.info
        """
        prior_factor_info = "\n".join(
            map(
                self._info_for_prior_factor,
                self.prior_factors
            )
        )
        analysis_factor_info = "\n\n".join(
            map(
                self._info_for_analysis_factor,
                self.analysis_factors
            )
        )
        return f"PriorFactors\n\n{prior_factor_info}\n\nAnalysisFactors\n\n{analysis_factor_info}"
