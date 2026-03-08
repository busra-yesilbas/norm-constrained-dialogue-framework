"""
Norm-Constrained Dialogue Framework
=====================================
A research simulator for evaluating AI alignment in ethically sensitive,
high-stakes conversational settings.

This package is intended strictly as an academic research sandbox.
All scenarios are synthetic and fictional. The framework does not
represent a production system, operational tool, or any real-world
deployment context.

Modules
-------
- config       : Configuration loading and management
- schemas      : Pydantic data models
- utils        : Shared utilities
- data         : Synthetic scenario generation
- agents       : Dialogue agent strategies
- simulation   : Respondent simulation and dialogue orchestration
- evaluation   : Norm compliance and utility evaluation
- experiments  : Multi-strategy comparison pipelines
- visualization: Plotting and reporting utilities
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from norm_dialogue_framework.config import load_config, FrameworkConfig

__all__ = ["load_config", "FrameworkConfig", "__version__"]
