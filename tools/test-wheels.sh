#!/bin/bash
set -e -u -x

# copy tests outside source to test the installed package and not from the repository
cp -r unit_tests /tmp/

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ "$PYBIN" == *cp2* || "$PYBIN" == *cp35* ]]; then
      # skip python2 and python3.5
      continue
    fi
    "$PYBIN/pip" install longitudinal_tomography -f wheelhouse/
    "$PYBIN/pip" install pytest pyyaml
    "$PYBIN/pytest" /tmp/unit_tests/
done
