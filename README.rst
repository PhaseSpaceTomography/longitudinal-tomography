.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/pipeline.svg
.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/coverage.svg
    :target: https://gitlab.cern.ch/anlu/longitudinaltomography/-/jobs/artifacts/master/download?job=pages

Copyright 2020 CERN. This software is distributed under the terms of the
GNU General Public Licence version 3 (GPL Version 3), copied verbatim in
the file LICENCE.txt. In applying this licence, CERN does not waive the
privileges and immunities granted to it by virtue of its status as an
Intergovernmental Organization or submit itself to any jurisdiction.


INSTALLATION
------------

The computationally intensive or time-critical parts of the library is
written in C++ and python bindings are provided using `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_.
The installation and usage of the library is the same for all operating systems, but
different dependencies are needed for different operating systems.

Prerequisites
=============

"""""
Linux
"""""

You need a C++ compiler like `g++` installed. This is not required if installing a prebuilt package from acc-py or pypi.

"""""""
Windows
"""""""

On Windows computers `MSVC >= 14.0 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools>`_
with the Windows 10 SDK is required.

In MinGW and WSL environments the standard `g++` compiler works out of the box.

"""""
MacOS
"""""

No offical tests have been done on MacOS, but presumably `g++`, `clang`/`llvm` should work.

Install
=======

The Longitudinal Tomography package is available in prebuilt wheels for Python 3.6-3.9
on CERN Acc-Py and pypy.org as `longitudinal-tomography`. The package can thus easily be installed on
a Linux machine using

::

    pip install longitudinal-tomography

The package can be installed on a MacOS or Windows machine in the same manner, but the
C++ extension will be built on install.

"""""""""""""""""""""
Other ways to install
"""""""""""""""""""""

Clone the repository and run
::

   pip install .

The C++ extension will be built on install.


For development environments where it's preferable to compile the C++ extension inplace, it's possible to run the command
::

    python setup.py build_ext --inplace

which will compile the C++ extension using the available compiler (decided by setuptools).

