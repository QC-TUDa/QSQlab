# QSQlab for the Quantum System Quizzing (QSQ) Certification Protocol

> A python library that uses Qiskit to simplify running the QSQ protocol on both quantum simulators and real
> hardware — from noise-model configuration, through sequence execution, to survival-probability
> analysis and plotting.

This project builds on Qiskit's `Backend` and `QuantumCircuit` classes to provide a complete,
batteries-included pipeline for single-qubit protocols: configure a
noisy simulator (or point at real hardware), run progressively longer gate sequences, and turn the
resulting counts into survival-probability plots and gate-fidelity estimates — with automatic,
crash-safe data logging along the way.

---

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Workflow](#workflow)
  - [1. Backends](#1-backends)
  - [2. `BenchmarkSequencer`](#2-benchmarksequencer)
  - [3. `PostProcessing`](#3-postprocessing)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Testing](#testing)
- [Project status & known limitations](#project-status--known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The codebase is organized around three stages that mirror how a certification experiment is actually
run:

1. **Define a backend** — either a `NoisyAerSimulator` with a configurable noise model, or a real
   hardware backend (ion-trap backends at Mainz University have been used during development).
2. **Run sequences** — `BenchmarkSequencer` builds and executes single-qubit circuits of increasing
   gate-sequence depth on that backend, collecting counts and metadata for each depth.
3. **Analyze results** — a `PostProcessing` subclass (e.g. `QSQPostProcessing`) turns the raw
   sequencer data into survival probabilities, binomial errors, and an optional plot.

Data flows between stages as plain `list[dict]` records or as JSON Lines (`.jsonl`) files, so any
stage can be run independently, re-run from saved data, or swapped out.


## Installation

This project is published to this repository's package registry. Install it with:

```bash
pip install qsqlab
```

> See this repository's **Packages** section for the exact package name and version tags.

Requires Python 3.10+ (the codebase uses `match` statements and `X | Y` union type hints
throughout) and a working Qiskit + Qiskit Aer installation.

## Quickstart

```python
import numpy as np

from qsqlab.backends import NoisyAerSimulator
from qsqlab.sequencer import BenchmarkSequencer
from qsqlab.data_processing import QSQPostProcessing
from qiskit import QuantumCircuit

# 1. Backend: a simulator with a small depolarizing + readout error
backend = NoisyAerSimulator(measurement_error_prob=0.01)
backend.add_depolarizing_error(param=0.005)

# 2. Sequencer: run an X-gate sequence at several depths, saving results as we go
sequencer = BenchmarkSequencer(backend=backend, qc=QuantumCircuit(1))
data = sequencer.run_circuits(
    depths=[0, 2, 4, 8, 16, 32],
    shots=2000,
    gate="rx",
    theta=np.pi/2
    save=True,
    savename="x_gate_run",
)

# 3. Post-processing: turn counts into survival probabilities and plot
result = QSQPostProcessing(sequencer_data=sequencer.data)
print(result)
print("ε map:", result._eps_map)
```

> Adjust the import paths above to match your installed package layout — see the
> [notebooks](#notebooks) for runnable, end-to-end examples.

## Workflow

### 1. Backends

#### `NoisyAerSimulator`

`NoisyAerSimulator` extends Qiskit Aer's `AerSimulator` with a friendly, incremental API for
composing noise models. Rather than constructing a `NoiseModel` by hand, you call one method per
error source; each call is validated, logged, and — for quantum (non-readout) errors — composed
into a running `SuperOp` exposed as `backend.gate_error`.

Supported error sources:

| Method | Error type |
|---|---|
| `add_readout_error` | Classical measurement (SPAM) error, symmetric or state-dependent |
| `add_depolarizing_error` | Depolarizing channel |
| `add_pauli_error` | Weighted Pauli (bit/phase-flip) channel |
| `add_amplitude_damping_error` | Amplitude damping (T1-like) |
| `add_phase_damping_error` | Phase damping (T2-like, dephasing only) |
| `add_phase_amplitude_damping_error` | Combined amplitude + phase damping |
| `add_thermal_relaxation_error` | Thermal relaxation from `t1`, `t2`, `gate_time` |
| `add_mixed_unitary_error` | Randomly-selected residual rotation, sampled per shot |
| `add_coherent_unitary_error` | Fixed, non-random residual rotation |

Every `add_*_error` method returns the constructed Qiskit Aer error object (or `None` if the
supplied parameter disables the error), and appends a matching `ErrorEntry` to
`backend.noise_log`. Because gate errors generally do **not** commute, `noise_log` preserves the
exact order errors were added in, and each `ErrorEntry` records that order — reconstructing a
`NoisyAerSimulator` from a saved log reproduces the same noise model, applied in the same sequence.

```python
sim = NoisyAerSimulator(measurement_error_prob=[0.01, 0.03])  # state-dependent readout error
sim.add_depolarizing_error(param=0.002)
sim.add_thermal_relaxation_error(t1=50e3, t2=30e3, gate_time=100)

print(sim)                 # human-readable noise log
print(sim.gate_error)      # composed SuperOp vs. identity, used for fidelity estimates
print(len(sim))            # number of registered errors
sim.clear_errors()         # reset to a clean AerSimulator-equivalent state
```

**`gate_error`** is the composition of the identity matrix with every quantum error added so far.
`BenchmarkSequencer` uses this directly to estimate the average gate fidelity contributed by the
simulator's noise model, independent of the specific circuit being run — see
[`_get_state_vector`](#2-benchmarksequencer) below.

#### Hardware backends

Any Qiskit `Backend`/`BackendV2` instance can be passed to `BenchmarkSequencer` in place of a
simulator. Ion-trap hardware backends from Mainz University have been the primary hardware target
during development; other Qiskit-compatible backends should work as long as they expose the
standard `.run(...)`/`.result()`/`.get_counts()` interface.

### 2. `BenchmarkSequencer`

`BenchmarkSequencer` builds single-qubit circuits of progressively increasing depth — repeating the
same gate `depth` times — runs them on a given backend, and records the outcome as a flat list of
result dictionaries.

```python
from qiskit import QuantumCircuit
sequencer = BenchmarkSequencer(backend=backend, qc=QuantumCircuit(1))

data = sequencer.run_circuits(
    depths=[0, 1, 2, 4, 8, 16],
    shots=1000,
    gate="x",
    measurement_basis="Z",
    save=True,            # write each depth's result to disk as soon as it's ready
    savepath="./data",
    savename="x_sequence",
)
```

Each result dictionary in `data` (and in the persisted `.jsonl` file) contains:

| Key | Description |
|---|---|
| `counts` | Measured counts from the backend |
| `ideal_counts` | Noiseless outcome probabilities (`'0'`/`'1'`) from statevector simulation |
| `gate` | Gate name used for this circuit |
| `angles` | `theta`/`phi`/`lam` values, if the gate is parametrized |
| `amount_of_gates` | Circuit depth |
| `shots` | Shots used |
| `measurement_basis` | `"X"`, `"Y"`, `"Z"`, or `None` |
| `gate_fid` | Estimated gate fidelity, or `None` if unavailable for this backend |
| `backend` | Class name of the backend used |
| `timestamp` / `unix_time` | When the circuit was run |

Supported gates include the standard single-qubit Qiskit gates (`x`, `y`, `z`, `h`, `s`, `sdg`,
`rx`, `ry`, `rz`, `p`, `u`, ...) plus three convenience aliases — `sqrtx`, `sqrty`, `sqrtz` —
implemented as `pi/2` rotations about X, Y, and Z respectively.

**Fidelity estimation.** `gate_fid` is computed differently depending on the backend:

- **Noisy `NoisyAerSimulator`** (has a populated `gate_error`): average gate fidelity between
  `gate_error` and the identity — the fidelity contributed by the configured noise model itself.
- **Noiseless `AerSimulator`-derived backend** (no `gate_error`): average gate fidelity between the
  executed and ideal circuits' unitary operators.
- **Any other backend** (real hardware): `gate_fid` is left as `None`, since neither of the above
  sources of ground truth is available for a physical device.

**Crash-safe by design.** Each depth's result is appended to `sequencer.data` and (if `save=True`)
flushed to the `.jsonl` file immediately after that depth completes — *before* the next depth
starts. If a run is interrupted partway through (a hardware failure, a lost connection, or a
manual stop), every depth that already finished is preserved on disk; nothing
from earlier in the run is lost if `save=True`.

**Optional transpilation.** Passing `transpile=True` runs each circuit through `qiskit.transpile`
for the target backend before execution, using `sequencer.transpiler_config` (see
[Configuration](#configuration)); per-call keyword arguments can override individual
pre-configured transpiler settings.

### 3. `PostProcessing`

`PostProcessing` (and its benchmark-specific subclasses, such as `QSQPostProcessing`) consumes
`BenchmarkSequencer` output — either a `list[dict]` passed directly, or a path to a saved `.jsonl`
file — and turns it into survival probabilities, binomial errors, and an optional plot.

```python
from qsqlab.data_processing import QSQPostProcessing

# From data already in memory:
result = QSQPostProcessing(sequencer_data=sequencer.data, show_plot=True)

# ...or from a previously saved run:
result = QSQPostProcessing(sequencer_data="data/x_sequence.jsonl", save=True, savefolder="figures")

# ...or from the most recently modified file in the default data directory:
result = QSQPostProcessing()

result.results          # BenchmarkResult(backend=..., gate=..., simulation=..., fidelity=..., depths=...)
result._eps_map    # dict[str, Any], one entry per depth
```

Key behavior:

- **Data loading** accepts a `list[dict]` (validated against the expected sequencer-record schema),
  a path to a `.jsonl` file, or `None` (in which case the most recently modified `.jsonl` file in
  the default data directory is used).
- **Simulation vs. hardware** is inferred automatically from the backend name against the
  `SIMULATOR_BACKENDS` config, or can be set explicitly with `simulation=True`/`False`.
- **Mixed-data protection**: if the supplied data spans more than one backend or more than one
  gate, `PostProcessing` raises `MixedDataException` rather than silently mixing incompatible
  results together.
- **Plotting** (`show_plot=True` and/or `save=True`) renders protocol-specific quantity vs. circuit
  length with binomial error bars, styled via `DEFAULT_PLOT_STYLE` (overridable per call), and can
  save both `.png` and `.pdf` copies of the figure.
- **Subclassing**: `PostProcessing` implements the shared plumbing (reading data, plotting,
  saving); gate-specific survival-probability logic (e.g. how "success" is defined for an `rz`
  sequence vs. an `rx`/`ry` sequence) belongs in a benchmark-specific subclass such as
  `QSQPostProcessing`.

## Configuration

Default behavior across the codebase is controlled by a handful of config modules, all overridable
per call or per instance:

| Module | Contents |
|---|---|
| `data_config` | `DEFAULT_DATA_ROOT` — default directory for saved `.jsonl` files and plots |
| `backend_config` | `SIMULATOR_BACKENDS` — backend names `PostProcessing` treats as simulation data |
| `gate_config` | `SINGLE_QUBIT_GATES`, `SU2` — gates/generators known to `BenchmarkSequencer` and `NoisyAerSimulator` |
| `plot_config` | `DEFAULT_PLOT_STYLE` — default `matplotlib.pyplot.rc_context` settings for plots |
| `transpiler_config` | `TRANSPILER_CONFIG` — default keyword arguments for `qiskit.transpile` |

## Notebooks

The `notebooks/` folder (outside `src/`) contains runnable examples of the full workflow,
including how to configure noisy simulations with `NoisyAerSimulator` and feed the results through
`BenchmarkSequencer` and `PostProcessing`. Start here if you'd rather learn by example than by
reading the API reference above.

## Testing

A `tests/` folder is included and covers the core classes. Test coverage is still being built out
as part of ongoing development — contributions of additional test cases are welcome.

## Project status & known limitations

This project is under early, active development. In particular:

- The current pipeline is scoped to **single-qubit** gate sequences; `BenchmarkSequencer` and
  `PostProcessing` assume one qubit throughout and do not yet validate multi-qubit input.
- APIs — especially `PostProcessing` and its subclasses — will change as the QSQ protocol's
  analysis requirements are refined.
- Please report bugs and unexpected behavior via this repository's issue tracker; feedback at this
  stage directly shapes the stable API for release.

## Contributing

Bug reports, feature requests, and pull requests are welcome — please open an issue describing
what you ran into, including the backend type, gate, and (if possible) a minimal reproducing
snippet.

## License

Licensed under the [Apache License 2.0](LICENSE).