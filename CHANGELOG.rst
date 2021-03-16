*********
Changelog
*********

master
======

v3.4.0
======

:Date: TBD

Fixes
-----

* Move usage of root logger in some modules to a named logger

New Features
------------

* Add `tomo.tracking.programs_machine.ProgramsMachine` to take voltage, phase and momentum arrays instead of a `.dat` file.
* Moved `tomo.data.data_treatment.make_phase_space` to C++
* Add `tomo.shortcuts` with macros for tracking and reconstruction.
* Add parent `TomoException` that all package exceptions inherit from to allow for easier try/catch
* Add optional callback functions in C++ tracking and reconstruct for progress tracking
* Overload C++ tracking to allow for phi12 passed as n_turns array instead of scalar

Other Changes
-------------

* Rewritten C++ library from Ctypes to PyBind11
    * Add project `CMakeLists.txt`
    * Platform independent extension compilation using `PyBind11Extension`
    * Same interface as ctypes wrapper, import from `tomo.cpp_routines.libtomo`
