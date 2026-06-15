"""Smoke test for the synthetic data generators.

Both Model A (constant frequency) and Model B (full 7-parameter timing model)
should build data that :func:`fermi_fold.fold` recovers with a decisive Q
statistic -- the same end-to-end check as ``test_fermi_fold.py``, but on data
generated from a known ground truth instead of the shipped ``data/data.npy``.

The generated lattices are small, so the recovered solutions have smaller phase
revolution counts than the real run; ``reasonable_k_std`` is lowered accordingly
(the reference runs reach Q ~ 2400, far above the threshold below).
"""

import pytest

from fermi_fold import fold
from model_a_constant_frequency import generate_data as generate_data_A
from model_b_full_timing import generate_data as generate_data_B

Q_DETECTION_THRESHOLD = 50.0
REASONABLE_K_STD = 1e3
GENERATORS = {"A": generate_data_A, "B": generate_data_B}


def _fold_robust(data, retries=6, **kwargs):
    """Run :func:`fold` with retries.

    g6k's shallow pump occasionally raises a (non-deterministic)
    ``SaturationError`` on the Model B lattice; retrying clears it.
    """
    last_exc = None
    for _ in range(retries):
        try:
            return fold(data, **kwargs)
        except Exception as exc:        # SaturationError is raised from inside g6k
            last_exc = exc
    raise last_exc


@pytest.mark.parametrize("model", ["A", "B"])
@pytest.mark.parametrize("fast", [False, True], ids=["full", "fast"])
def test_generated_data_round_trips(model, fast):
    data = GENERATORS[model]()

    # Model B's shallow pump saturates more reliably a touch shallower than the
    # default (27); full mode is unaffected.
    fold_kwargs = {"reasonable_k_std": REASONABLE_K_STD}
    if fast and model == "B":
        fold_kwargs["pump_stop"] = 30

    result = _fold_robust(data, fast=fast, **fold_kwargs)

    mask = result["reasonable_solutions_mask"]
    assert mask.any(), f"model {model}: no physically reasonable solutions found"

    max_q = result["Q_stat"][mask].max()
    assert max_q > Q_DETECTION_THRESHOLD, (
        f"model {model} ({'fast' if fast else 'full'}): max Q {max_q:.1f} is "
        f"below the detection threshold {Q_DETECTION_THRESHOLD}"
    )
