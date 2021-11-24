*********
Changelog
*********

master
======

v3.4.1
------

:Date: 2021-11-24

-----
Fixes
-----

* Increase number of interpolated turns for :code:`ProgramsMachine`.

------------
New Features
------------

* add :code:`set_num_threads` to :code:`libtomo` extension to limit max number of threads allowed to be used by OpenMP parallelizations.

v3.4.0
------

:Date: 2021-07-22

-----
Fixes
-----

* Move usage of root logger in some modules to a named logger.
* Add missing typing hints.
* Cleanup of :code:`MANIFEST.in`.
* Updated ``README`` with full installation instructions.
* Fix Example 00 that broke after updating the tomoscope interface.
* Fixed an issue in the normalize function of reconstruct.cpp that would cause extension compiled with debug symbols (-g) to have inconsistent normalization results due to a race condition.
* Fixed an issue where locally allocated memory in the reconstruct function would not be freed if an exception was thrown mid-function.

------------
New Features
------------

* Add :code:`longitudinal_tomography.tracking.programs_machine.ProgramsMachine`
    * Takes voltage, phase and momentum arrays instead of linearly extrapolating using derivatives. This replaces :code:`vrf1, vrf1dot, vrf2, vrf2dot, b0, bdot` in the original :code:`Machine`.
    * Creation of a superclass :code:`MachineABC` that :code:`Machine` and :code:`ProgramsMachine` inherit from. Relevant functions overloaded with :code:`multipledispatch` has been updated accordingly.
    * The majority of the keyword arguments previously processed in :code:`Machine` have been moved to the superclass.
    * Documentation from the original :code:`Machine` has been split between it and the superclass.
    * The :code:`longitudinal_tomography.utils.physics` module has been updated to provide objective functions for the :code:`ProgramsMachine` for Newton optimization.
    * Some functions using :code:`physics.vrft` to calculate voltage at turn using derivatives have been changed to use the :code:`vrf_at_turn` arrays instead for compatibility with the new :code:`ProgramsMachine`.
* Moved :code:`longitudinal_tomography.data.data_treatment.make_phase_space` to C++ to use a tight for-loop instead of a python for-zip loop.
* Add :code:`longitudinal_tomography.shortcuts` with macros for tracking and reconstruction.
* Add parent :code:`TomoException` that all custom exceptions inherit from to allow for easier try/catch when using the API.
* Add optional passing of callback functions in C++ tracking and reconstruct for progress tracking
* Overload C++ tracking to allow for phi12 passed as n_turns array instead of scalar
* The :code:`longitudinal_tomography.utils.tomo_output.show` function now has an optional :code:`figure` argument so the user can pass a pre-created figure.

-------------
Other Changes
-------------

* Rewritten C++ library from Ctypes to PyBind11
    * PyBind11 allows for passing native Python types to C++ and interaction with them, and specifically supports direct access to the underlying NumPy array data pointers. The performance remains the same as with the old ctypes extension.
    * The extension wrappers are now on the C side, instead of on the Python side. However the interface remains the same, the extension is imported as :code:`longitudinal_tomography.cpp_routines.libtomo`.
    * All usages of the old Python wrappers have been replaced by the PyBind11 extension.
    * Documentation for the module is available in the same way as a python module as :code:`help(longitudinal_tomography.cpp_routines.libtomo)`.
    * Headers for all source files have been added.
    * Compilation
        * The :code:`setup.py` has been updated for the necessary compilation. The compilation should now be natively supported on all operating systems.
        * The :code:`__restrict__` specifier has been removed to allow for compilation using MSVC on Windows.
        * Add project :code:`CMakeLists.txt` that may be used as project file in CLion, but does currently not correctly install the extension.

* The calculation of `rms dp/p` no longer requires passing the momentum value as this can be calculated using the energy and mass.

------------
Deprecations
------------
* With the introduction of the PyBind11 extension, the python-side C wrappers at :code:`longitudinal_tomography.cpp_routines.tomolib_wrappers` are deprecated. The same functions that existed in the :code:`tomolib_wrappers` can instead be imported from :code:`longitudinal_tomography.cpp_routines.libtomo`.
* The function :code:`longitudinal_tomography.data.data_treatment._make_phase_space` has moved to C++ and should be imported from :code:`longitudinal_tomography.cpp_routines.libtomo` if necessary.
* Removal of :code:`longitudinal_tomography.compile` that was used to compile the legacy ctypes extension.
* Removal of :code:`longitudinal_tomography.tomoscope_interface` since the same functionality is provided in the :code:`longitudinal_tomography.__main__` entrypoint.

v3.3.2
------

:Date: 2021-02-10

-----
Fixes
-----

* Corrected saving of profile to the recreated instead of the original waterfall.

v3.3.1
------

:Date: 2021-02-10

-----
Fixes
-----

* Corrected indexing of save profile.
* Changed :code:`Machine.filmstop` to correct value.

------------
New Features
------------

* Added :code:`save_profile` function to :code:`longitudinal_tomography.compat.tomoscope` module.

v3.3.0
------

:Date: 2021-02-09

---------------
BREAKING CHANGE
---------------

* Renamed :code:`tomo` module to :code:`longitudinal_tomography` for compatibility with Acc-Py.

v3.2.0
------

:Date: 2021-01-19

------------
New Features
------------

* Addition of a :code:`__main__` entrypoint to the :code:`tomo` package that provides the same functionality as the :code:`tomoscope_interface` script and :code:`run.py` in the root of the repository. This allows for a reconstruction to be run with simply :code:`python -m tomo [args]` or :code:`acc-py app run tomo [args]`. For detailed usage, execute :code:`python -m tomo --help` or :code:`acc-py app run tomo --help`.
* Addition of :code:`tomo.compat.tomoscope` submodule that houses tomoscope-specific I/O functions.

-------------
Other Changes
-------------

* Adapted :code:`tomo.utils.tomo_input` and :code:`tomo.utils.tomo_run` for the new :code:`__main__` entrypoint.

------------
Deprecations
------------

* Removed :code:`tomoscope_interface` and :code:`run.py` as they have been replaced by :code:`__main__`.

v3.1.0
------

:Date: 2021-01-07

-----
Fixes
-----

* Initialize all class member variables at :code:`__init__` and checking against :code:`None` instead of with :code:`__hasattr__`.
* Optimized :code:`tomo.particles.physical_to_coords` and removed unnecessary array slicing.
* Fixed a memory leak in the reconstruction function that did not properly free allocated memory.
* Refactored :code:`exceptions` and :code:`assertions` to package root to avoid circular imports.
* Add :code:`dEbin` and :code:`weight` as attributes to :code:`Machine` and :code:`Tomography` classes respectively for consistency.
* :code:`tomo.data.data_treatment.rebin` will now only return rebinned :code:`waterfall` and :code:`dtbin` if :code:`synch_part_x` was not passed.

------------
New Features
------------

* Added :code:`setup.cfg`, :code:`pyproject.toml` and :code:`setup.py` to enable installation with :code:`pip`.
* Refactored legacy functions that enable interfacing with legacy fortran io to :code:`tomo.compat` module.
* Added some imports to :code:`__init__` files for easier imports. For instance `Machine` can now be imported from :code:`tomo.tracking` directly.
* Added typing hints to most of the code.
* Add a :code:`tomoscope_interface` script to the root of the package that serves as an entrypoint for the Tomoscope.
* Addition of a :code:`tomo.data.pre_process` module that houses functions for pre-processing of raw data (waterfalls). Some functions from :code:`tomo.data.data_treatment` were moved.
* Addition of a :code:`tomo.data.post_process` module that houses functions for calculation of RMS and 90% emittance as well as RMS dp/p. The functions are overloaded using :code:`multipledispatch`.

-------------
Other Changes
-------------

* Corrected spelling in symbols and documentation.
* Corrected code for PEP8 compliance.
* Implementation of a full CI test->release pipeline.
    * Added scripts to test and build python wheels in a :code:`manylinux2014` docker image.
    * Re-pointed coverage badge URLs to main Gitlab repository instead of a fork.
    * Created Gitlab CI pipeline to test code, build and test wheels and source distributions, and release on tag.
    * Adapted CI build stage to Acc-Py wheel building, and changed CI base image to :code:`python:3.6` to avoid having to install conda.
    * Release wheels and source distributions to Acc-Py, PyPI and TestPyPI on tag.

------------
Deprecations
------------

* Python 3.5 is no longer supported as type hinting was introduced in Python 3.6.
* Removed legacy :code:`compile.py` for compiling the C++ extension. The extension can now be built using :code:`pip` or :code:`setup.py`.
* Removed :code:`-march=native` from compile options.

v3.0.0
------

Initial release as Python package. See CERN ATS note for a detailed description.
