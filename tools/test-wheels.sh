#!/bin/bash
set -e -x

# copy tests outside source to test the installed package and not from the repository
cp -r unit_tests /tmp/

echo "Testing install of longitudinal_tomography wheels for version $VERSION"

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ "$PYBIN" == *cp2* || "$PYBIN" == *cp35* ]]; then
      # skip python2 and python3.5
      continue
    fi
    if [[ -z $CI_COMMIT_TAG ]]; then
      VERSION=`$PYBIN/python -c "import longitudinal_tomography as tomo; print(tomo.__version__)"`
    else
      VERSION=$CI_COMMIT_TAG
    fi
    "$PYBIN/pip" install longitudinal_tomography==$VERSION -f wheelhouse/
    "$PYBIN/pip" install pytest pyyaml
    "$PYBIN/pytest" /tmp/unit_tests/
done
