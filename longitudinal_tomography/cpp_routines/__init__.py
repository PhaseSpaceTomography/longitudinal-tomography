"""Compiled C++ routines."""
from __future__ import annotations

import importlib
import importlib.machinery as machinery
import importlib.util as util
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_NAME = f'{__name__}.libtomo'
_HERE = os.path.realpath(os.path.dirname(__file__))


def _installed_libtomo() -> ModuleType:
    """Load libtomo from a copy of this package elsewhere on ``sys.path``.

    An uncompiled source directory shadows an installed package whenever it
    comes first on the path, so search the remaining entries for the
    extension.
    """
    parts = __name__.split('.')
    search = []
    for entry in sys.path:
        directory = os.path.realpath(os.path.join(entry or os.curdir, *parts))
        if directory != _HERE and os.path.isdir(directory):
            search.append(directory)

    spec = machinery.PathFinder.find_spec(_NAME, search)
    if spec is None:
        raise ImportError(
            f'no compiled {_NAME} extension in {_HERE} or on sys.path; '
            f'install the package to build it')

    module = util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
    return module


try:
    libtomo = importlib.import_module(_NAME)
except ModuleNotFoundError as exc:
    # A dependency missing inside libtomo raises this too; only the extension
    # itself being absent warrants the search.
    if exc.name != _NAME:
        raise
    libtomo = _installed_libtomo()
