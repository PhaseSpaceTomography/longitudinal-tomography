.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/pipeline.svg
.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/coverage.svg
    :target: https://gitlab.cern.ch/anlu/longitudinaltomography/-/jobs/artifacts/master/download?job=pages

Copyright 2020 CERN. This software is distributed under the terms of the
GNU General Public Licence version 3 (GPL Version 3), copied verbatim in
the file LICENCE.txt. In applying this licence, CERN does not waive the
privileges and immunities granted to it by virtue of its status as an
Intergovernmental Organization or submit itself to any jurisdiction.


INSTALL
-------

Clone the repository and run
::

   pip install .


For development environments where it's preferable to compile the C++ extension inplace, it's possible to run the command
::

    python setup.py build_ext --inplace

which will compile the C++ extension using the available compiler (decided by setuptools).


Requirements
------------

1. A gcc compiler with C++11 support (version greater than 4.8.4).  

2. An Anaconda distribution (Python 3 recommended).

3. That's all!
