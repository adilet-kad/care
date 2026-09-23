"""care.conformal -- the conformal risk controller + Mondrian strata."""

from care.conformal.bounds import clopper_pearson_upper, hoeffding_upper
from care.conformal.combine import Combiner, MeanCombiner
from care.conformal.controller import ConformalController
from care.conformal.rcps import rcps_threshold
from care.conformal.strata import (
    DEFAULT_STRATUM, by_cardinality, by_column, by_domain, by_product, by_source,
    cardinality_domains, marginal_strata,
)

__all__ = [
    "ConformalController", "rcps_threshold", "hoeffding_upper", "clopper_pearson_upper",
    "Combiner", "MeanCombiner",
    "DEFAULT_STRATUM", "marginal_strata", "by_column", "by_domain", "by_source",
    "by_cardinality", "by_product", "cardinality_domains",
]
