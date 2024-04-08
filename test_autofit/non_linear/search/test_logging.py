import logging

import autofit as af


class Analysis(af.mock.MockAnalysis):
    def log_likelihood_function(self, instance):
        logging.warning("Loud logging")
        return super().log_likelihood_function(instance)


def test_logging():
    search = af.mock.MockSearch("name", fit_fast=False)
    analysis = Analysis()
    model = af.Model(af.Gaussian)
    search.paths.remove_files = False
    search.fit(model, analysis)
    assert search.paths.output_path.exists()
    assert (search.paths.output_path / "search.log").exists()
