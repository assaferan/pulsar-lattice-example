# pulsar-lattice-example

[![CI](https://github.com/assaferan/pulsar-lattice-example/actions/workflows/ci.yml/badge.svg)](https://github.com/assaferan/pulsar-lattice-example/actions/workflows/ci.yml)

This repository contains an example of recovering the periodicity of PSR J0318+0253 (4FGL J0318.2+0254) using the lattice periodicity search approach described in [Gazith et al. 2025](https://ui.adsabs.harvard.edu/abs/2025ApJ...979...48G/abstract).

The Data was reduced from the Fermi-LAT database using fermitools, and some dedicated auxiliary functions.

Use the requirements file to set up the Python environment, although to install *g6k*, you might want to consult its respective [repository](https://github.com/fplll/g6k).

## Creating an environment
```
git clone https://github/fplll/g6k
cd g6k
conda create --name g6k
conda activate g6k
conda install numba
pip install -r requirements.txt
conda install -c conda-forge fpylll
python setup.py build_ext --inplace
python -m pytest
```

### Building g6k from source (no conda)

On a Debian/Ubuntu machine, g6k's own `bootstrap.sh` builds `fplll`, `fpylll`
and `g6k` into a self-contained virtualenv. First install the build
dependencies:
```
sudo apt-get install -y build-essential libgmp-dev libmpfr-dev \
    libtool-bin libqd-dev autoconf automake pkg-config \
    python3-pip python3-virtualenv
```
Then build (this also creates the `g6k-env` virtualenv):
```
git clone https://github.com/fplll/g6k
cd g6k
PYTHON=python3 ./bootstrap.sh -j 8
```
Activate the environment in each new shell before running anything that imports
`fpylll`/`g6k`:
```
source /path/to/g6k/activate
```

## Running the folding pipeline

[`fermi_fold.py`](fermi_fold.py) packages the notebook into an importable
`fold()` function. With the g6k environment active:
```
python3 fermi_fold.py            # folds data/data.npy and prints the max Q statistic
```
or from your own code:
```python
from fermi_fold import load_data, fold

data = load_data("data/data.npy")
result = fold(data)
best = result["verify_fold"][result["reasonable_solutions_mask"]][
    result["Q_stat"][result["reasonable_solutions_mask"]].argmax()
]
```

[`run.sh`](run.sh) is a convenience wrapper that activates the g6k environment
and runs the pipeline, so you don't have to `source` it manually:
```
./run.sh                         # runs fermi_fold.py
./run.sh some_other_script.py    # runs any script inside the g6k env
```
If your g6k checkout is not at `~/g6k`, point the wrapper at it:
```
G6K_ACTIVATE=/path/to/g6k/activate ./run.sh
```
