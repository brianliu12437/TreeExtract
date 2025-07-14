"""
code package
============

This file marks the *code* directory as a Python package and makes everything
defined in *methods.py* directly importable from the package level:

    from code import some_function

You can also access the module itself with:

    import code.methods as methods
"""
from .methods import *  # re‑export all public names from methods

__all__ = [name for name in globals() if not name.startswith("_")]
__version__ = "0.1.0"
