from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from . import commons

if TYPE_CHECKING:
    from longitudinal_tomography.data.profiles import Profiles
    from longitudinal_tomography.tracking import Machine
    from longitudinal_tomography.utils.tomo_input import Frames


@pytest.fixture(scope='session')
def machine_frames_profiles() -> tuple[Machine, Frames, Profiles]:
    """Load and track INDIVShavingC325.dat once for the whole test run."""
    return commons.load_data()
