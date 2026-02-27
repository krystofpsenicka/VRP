# RAPIDS Solver Environment Setup

The GPU-accelerated distance matrix computation (cuGraph Dijkstra) and the
VRP solver (NVIDIA cuOpt) run inside a dedicated conda environment called
**`rapids_solver`** that is isolated from the main cuRobo / Isaac Sim
environment.

---

## 1. Prerequisites

| Requirement | Version |
|-------------|---------|
| CUDA Toolkit | ≥ 12.0 |
| Conda / Mamba | any recent |
| GPU with CC ≥ 7.0 (Volta+) | — |

---

## 2. Create and populate the environment

```bash
# Create a fresh Python 3.11 environment
conda create -n rapids_solver python=3.11 -y
conda activate rapids_solver

# Install RAPIDS (cuGraph + cuDF + cuPy) from NVIDIA's channel
pip install \
  --extra-index-url https://pypi.nvidia.com \
  cugraph-cu12 \
  cudf-cu12 \
  cupy-cuda12x

# Install cuOpt (NVIDIA's GPU VRP solver)
pip install \
  --extra-index-url https://pypi.nvidia.com \
  cuopt-cu12

# Install OR-Tools as a CPU fallback (same env)
pip install ortools

# Verify
python -c "import cugraph; print('cugraph OK')"
python -c "from cuopt import routing; print('cuOpt OK')"
```

> **Tip:** If `pip install cuopt-cu12` fails due to network restrictions,
> download the wheel from [pypi.nvidia.com](https://pypi.nvidia.com) and
> install it offline:
> ```bash
> pip install cuopt_cu12-*.whl
> ```

---

## 3. Configuring the environment path

The planner locates the `rapids_solver` Python interpreter via the
`RAPIDS_PYTHON` setting in `VRP/config.py`:

```python
# VRP/config.py
RAPIDS_PYTHON = os.environ.get(
    "RAPIDS_PYTHON",
    os.path.expanduser("~/miniconda3/envs/rapids_solver/bin/python"),
)
```

Override with an environment variable if your conda root differs:

```bash
export RAPIDS_PYTHON="/opt/conda/envs/rapids_solver/bin/python"
python VRP/run_vrp.py --num_robots 2
```

Or pass it directly on the CLI:

```bash
python VRP/run_vrp.py \
  --num_robots 2 \
  --rapids_python /opt/conda/envs/rapids_solver/bin/python
```

---

## 4. Verify the full subprocess path

```bash
# From the main (cuRobo) environment:
python -c "
from VRP.gpu_distance_matrix import _rapids_env_has_cugraph
print('cuGraph available in rapids_solver:', _rapids_env_has_cugraph())
"
```

Expected output:
```
cuGraph available in rapids_solver: True
```

---

## 5. Fallback behaviour

| Stage | Primary (GPU) | Fallback (CPU) |
|-------|--------------|----------------|
| Distance matrix | cuGraph Dijkstra in `rapids_solver` | A* on CPU (NumPy heapq) |
| VRP solve | cuOpt in `rapids_solver` | OR-Tools `GUIDED_LOCAL_SEARCH` |
| Trajectory planning | cuRobo `num_seeds=1024` | OMPL RRT* warm-start |

All fallbacks activate automatically – no manual intervention required.

---

## 6. Quick smoke test

```bash
# Activate the MAIN environment (cuRobo / Isaac Sim env), then:
cd /home/troja_robot_lab/Desktop/Krystof
python VRP/run_vrp.py \
  --num_robots 2 \
  --random_waypoints 4 \
  --solver auto \
  --verbose
```

This will:
1. Build (or load cached) occupancy grid from the wreck mesh.
2. Sample 4 random collision-free waypoints.
3. Compute the 4×4 GPU distance matrix via cuGraph subprocess.
4. Solve the 2-vehicle VRP with cuOpt (fallback to OR-Tools).
5. Resolve traffic-light conflicts.
6. Plan trajectories with cuRobo (fallback to RRT*).
7. Print per-robot summary.
