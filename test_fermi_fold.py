"""Smoke test for the lattice folding pipeline.

Runs :func:`fermi_fold.fold` on the shipped data and checks that it recovers
the pulsar, i.e. that some physically reasonable solution has a Q statistic
far above the null population.  The reference run reaches Q ~= 400; the
threshold is set well below that to tolerate the randomness in sieving while
still requiring a clear detection.
"""

from fermi_fold import load_data, fold

Q_DETECTION_THRESHOLD = 50.0


def test_fold_detects_pulsar():
    data = load_data("data/data.npy")
    result = fold(data)

    mask = result["reasonable_solutions_mask"]
    assert mask.any(), "no physically reasonable solutions were found"

    max_q = result["Q_stat"][mask].max()
    assert max_q > Q_DETECTION_THRESHOLD, (
        f"max Q statistic {max_q:.1f} is below the detection threshold "
        f"{Q_DETECTION_THRESHOLD}; the pulsar was not recovered"
    )
