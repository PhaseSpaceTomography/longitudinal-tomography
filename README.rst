===================================
Longitudinal Phase Space Tomography
===================================

.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/pipeline.svg
.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/coverage.svg
    :target: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/-/jobs

Copyright 2025 CERN. This software is distributed under the terms of the
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

Installing using package manager
""""""""""""""""""""""""""""""""

The Longitudinal Tomography package is available in prebuilt wheels for Python 3.11
on CERN Acc-Py and pypy.org as `longitudinal-tomography`. The package can thus easily be installed on
a Linux machine using

::

    pip install longitudinal-tomography

The package can be installed on a MacOS or Windows machine in the same manner, but the
C++ extension will be built on install.


Installing manually
"""""""""""""""""""

Prerequisites
~~~~~~~~~~~~~

**Linux**

You need a C++ compiler like `g++` installed. This is not required if installing a prebuilt package from acc-py or pypi.

**Windows**

On Windows computers `MSVC >= 14.0 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools>`_
with the Windows 10 SDK is required.

In MinGW and WSL environments the standard `g++` compiler works out of the box.

**MacOS**

You need to use a compiler other that the default provided on MacOS (:code:`gcc` is symlinked to :code:`clang` by default).
The easiest way (and the way that us currently supported) is to install :code:`llvm` and :code:`openmp` with Homebrew: :code:`brew install llvm openmp`.

Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

For MacOS see next section.

Clone the repository and run
::

   pip install .

The C++ extension will be built on install using the native compiler in Linux and Windows (pybind11 should find it).


For development environments where it's preferable to compile the C++ extension inplace, it's possible to run the command
::

    pip install -e .

which will compile the C++ extension using the available compiler (decided by setuptools).

**MacOS**

MacOS requires some special treatment for the extension to compile.
You need to tell pip to use Homebrew `clang` and `llvm` instead of the default
compiler and libraries. This can be done by setting the `CC`, `LDFLAGS` and `CPPFLAGS` environment variables.
You also need to install the `libomp` package with Homebrew.

On arm64 (M1/M2) MacBooks, use the following:
..

    export LDFLAGS="-L/opt/homebrew/lib -L/opt/homebrew/opt/llvm/lib"
    export CPPFLAGS="-I/opt/homebrew/include -I/opt/homebrew/opt/llvm/include"
    CC=/opt/homebrew/opt/llvm/bin/clang++
    pip install .

Or use the :code:`-e` flag for an editable installation.
For Intel MacBooks, Homebrew packages are installed in a different location.

Hence, for Intel MacBooks, use the following:
::

    export LDFLAGS="-L/usr/local/opt/homebrew/lib -L/usr/local/opt/homebrew/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/homebrew/include -I/usr/local/opt/homebrew/opt/llvm/include"
    CC=/usr/local/opt/llvm/bin/clang++ pip install .

"""""""""""""
Documentation
"""""""""""""

This development is based on the well tested and widely used FORTRAN95 code, documented and available here: http://tomograp.web.cern.ch/tomograp/
Details on the algorithms in both codes, and the differences between them, can be found here: https://cdsweb.cern.ch/record/2750116?ln=ka

"""""""""""""
Documentation
"""""""""""""

This development is based on the well tested and widely used FORTRAN95 code, documented and available here: http://tomograp.web.cern.ch/tomograp/
Details on the algorithms in both codes, and the differences between them, can be found here: https://cdsweb.cern.ch/record/2750116?ln=ka


Parallelization using OpenMP

The C++ extension is accelerated by OpenMP parallel for loops. It is possible to limit the number of launched threads
by setting it in the extension, by
::

    from longitudinal_tomography.cpp_routines import libtomo
    libtomo.set_num_threads([num_threads])

which will set the maximum number of used threads to :code:`[num_threads]`.
