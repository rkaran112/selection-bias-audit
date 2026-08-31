"""Pluggable reject-inference methods.

Importing this package registers all four in `base.REGISTRY`.
"""
from .base import (InferenceContext, InferenceResult, REGISTRY, register,
                   run_all)
from . import parcelling, fuzzy, ipw, heckman   # noqa: F401  (registration)

METHOD_ORDER = ["parcelling", "fuzzy", "ipw", "heckman"]
