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

* Move usage of root logger in some modules to a named logger.
* Add missing typing hints.
* Cleanup of ``MANIFEST.in``.
* Updated ``README`` with full installation instructions.
* Fix Example 00 that broke after updating the tomoscope interface.

New Features
------------

* Add :code:`tomo.tracking.programs_machine.ProgramsMachine`
    * Takes voltage, phase and momentum arrays instead of linearly extrapolating using derivatives. This replaces :code:`vrf1, vrf1dot, vrf2, vrf2dot, b0, bdot` in the original :code:`Machine`.
    * Creation of a superclass :code:`MachineABC` that :code:`Machine` and :code:`ProgramsMachine` inherit from. Relevant functions working with :code:`multipledispatch` has been updated accordingly.
    * The majority of the keyword arguments previously processed in :code:`Machine` have been moved to the superclass.
    * Documentation from the original :code:`Machine` has been split between it and the superclass.
    * The :code:`tomo.utils.physics` module has been updated to provide objective functions for the :code:`ProgramsMachine` for Newton optimization.
    * Some functions using :code:`physics.vrft` to calculate voltage at turn using derivatives have been changed to use the `vrf_at_turn` arrays instead for compatibility with the new :code:`ProgramsMachine`.
* Moved :code:`tomo.data.data_treatment.make_phase_space` to C++ to use a tight for-loop instead of a python for-zip loop.
* Add :code:`tomo.shortcuts` with macros for tracking and reconstruction.
* Add parent :code:`TomoException` that all custom exceptions inherit from to allow for easier try/catch when using the API.
* Add optional passing of callback functions in C++ tracking and reconstruct for progress tracking
* Overload C++ tracking to allow for phi12 passed as n_turns array instead of scalar
* The :code:`tomo.utils.tomo_output.show` function now has an optional :code:`figure` argument so the user can pass a pre-created figure.

Other Changes
-------------

* Rewritten C++ library from Ctypes to PyBind11
    * PyBind11 allows for passing native Python types to C++ and interaction with them, and specifically supports direct access to the underlying NumPy array data pointers. The performance remains the same as with the old ctypes extension.
    * The extension wrappers are now on the C side, instead of on the Python side. However the interface remains the same, the extension is imported as :code:`tomo.cpp_routines.libtomo`.
    * All usages of the old Python wrappers have been replaced by the PyBind11 extension.
    * Documentation for the module is available in the same way as a python module as :code:`help(tomo.cpp_routines.libtomo)`.
    * Headers for all source files have been added.
    * Compilation
        * The :code:`setup.py` has been updated for the necessary compilation. The compilation should now be natively supported on all operating systems.
        * The :code:`__restrict__` specifier has been removed to allow for compilation using MSVC on Windows.
        * Add project :code:`CMakeLists.txt` that may be used as project file in CLion, but does currently not correctly install the extension.
* The calculation of `rms dp/p` no longer requires passing the momentum value as this can be calculated using the energy and mass.

Deprecations
------------
* With the introduction of the PyBind11 extension, the python-side C wrappers at :code:`tomo.cpp_routines.tomolib_wrappers` are deprecated. The same functions that existed in the :code:`tomolib_wrappers` can instead be imported from :code:`tomo.cpp_routines.libtomo`.
* The function :code:`tomo.data.data_treatment._make_phase_space` has moved to C++ and should be imported from :code:`tomo.cpp_routines.libtomo` if necessary.
* Removal of :code:`tomo.compile` that was used to compile the legacy ctypes extension.
* Removal of :code:`tomo.tomoscope_interface` since the same functionality is provided in the :code:`tomo.__main__` entrypoint.
