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

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "$PYBIN" == *cp2* || "$PYBIN" == *cp35* ]]; then
      # skip python2 & python3.5
      continue
    fi
    "${PYBIN}/pip" wheel . --no-deps -w build
done

# Bundle external shared libraries into the wheels
for whl in build/*.whl; do
    repair_wheel "$whl"
done
