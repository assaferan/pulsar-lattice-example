"""Synthetic data generators for the lattice folding pipeline.

The pipeline (``fermi_fold.fold``) recovers a pulsar timing solution by finding
a short vector in a lattice built from photon arrival times.  These generators
build synthetic data dicts with the same keys as ``data/data.npy`` so the whole
pipeline can be exercised end-to-end on a known ground truth.

Two models, in separate files for reference:

* ``model_a_constant_frequency`` -- Model A: a constant-frequency pulsar, two
  parameters ``(phi, f)``.  The smallest end-to-end example.
* ``model_b_full_timing``        -- Model B: the full 7-parameter timing model
  ``(phi, f, fdot, dalpha, ddelta, dalpha_dot, ddelta_dot)``, with the spin
  polynomial plus astrometry via the Roemer delay.

Each module exposes ``generate_data(...)`` and is runnable on its own::

    python model_a_constant_frequency.py   # -> data/data_generated_A.npy
    python model_b_full_timing.py          # -> data/data_generated_B.npy

For convenience they are re-exported here:

    from data_generation import generate_data_A, generate_data_B
"""

from model_a_constant_frequency import generate_data as generate_data_A
from model_b_full_timing import generate_data as generate_data_B

__all__ = ["generate_data_A", "generate_data_B"]
