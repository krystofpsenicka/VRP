#!/usr/bin/env bash
# setup_rapids_env.sh
#
# Creates the `rapids_solver` conda environment with RAPIDS 25.04 (cuGraph +
# cuOpt) for GPU-accelerated VRP solving.
#
# Requirements:
#   - CUDA 12.x driver (already confirmed: 575.64.03 / CUDA 12.9)
#   - RTX 3090 (sm_86, Ampere – fully supported by RAPIDS 25.04)
#   - conda / miniconda in PATH
#
# Usage:
#   bash VRP/setup_rapids_env.sh
#   # Takes ~15–45 min depending on connection (downloads ~5–8 GB)
#
# After creation, run_vrp.py auto-detects the env via config._find_rapids_python().

set -euo pipefail

ENV_NAME="rapids_solver"
RAPIDS_VER="25.04"
PYTHON_VER="3.11"
CUDA_SPEC="cuda-version>=12.0,<=12.9"

echo "=== Creating conda env: ${ENV_NAME} (RAPIDS ${RAPIDS_VER}, Python ${PYTHON_VER}) ==="

# Remove any failed partial install
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "  Found existing '${ENV_NAME}' env – removing it first…"
    conda env remove -n "${ENV_NAME}" -y
fi

conda create -n "${ENV_NAME}" \
    -c rapidsai -c conda-forge -c nvidia \
    "rapids=${RAPIDS_VER}" \
    "python=${PYTHON_VER}" \
    "${CUDA_SPEC}" \
    -y

echo ""
echo "=== Installing cuOpt ==="
# cuOpt may ship via pip wheel for RAPIDS 24+
conda run -n "${ENV_NAME}" pip install "cuopt-cu12" --extra-index-url https://pypi.nvidia.com || {
    echo "  pip install failed – trying conda channel…"
    conda install -n "${ENV_NAME}" \
        -c rapidsai -c conda-forge -c nvidia \
        "cuopt" -y || echo "  cuOpt not in conda channel – skipping (cuGraph still available)"
}

echo ""
echo "=== Verifying installation ==="
conda run -n "${ENV_NAME}" python - <<'PYEOF'
import sys
print(f"Python:  {sys.version}")

try:
    import cudf, cugraph
    print(f"cuDF:    {cudf.__version__}")
    print(f"cuGraph: {cugraph.__version__}")
except Exception as e:
    print(f"  cuDF/cuGraph import error: {e}")

try:
    from cuopt.routing import DataModel, SolverConfig
    print("cuOpt:   DataModel API (>=24.x)  ✓")
except ImportError:
    try:
        from cuopt import routing
        print("cuOpt:   Solver API (<=23.x)  ✓")
    except ImportError as e:
        print(f"  cuOpt not available: {e}")

PYEOF

echo ""
echo "=== Done! ==="
echo "rapids_solver env is ready.  run_vrp.py will use it automatically."
echo "To test GPU VRP: conda run -n rapids_solver python VRP/cuopt_subprocess.py"
