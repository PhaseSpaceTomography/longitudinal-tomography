#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w wheelhouse/
    fi
}


# Install a system package required by our library
yum install -y gcc

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "$PYBIN" == *cp2* || "$PYBIN" == *cp35* ]]; then
      # skip python2
      continue
    fi
    "${PYBIN}/pip" wheel . -w build
done

# Bundle external shared libraries into the wheels
for whl in build/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ "$PYBIN" == *cp2* || "$PYBIN" == *cp35* ]]; then
      # skip python2
      continue
    fi
    "${PYBIN}/pip" install longitudinal_tomography --no-index -f build/
done
