# QSQlab for the Quantum System Quizzing (QSQ) Certification Protocol

> A python library that uses Qiskit to simplify running the [QSQ protocol](https://doi.org/10.48550/arXiv.2608.17005) on both quantum simulators and real
> hardware — from noise-model configuration, through sequence execution, to data
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
    - [`BenchmarkResult`](#benchmarkresult)
  - [4. `QSQPostProcessing`](#4-qsqpostprocessing)
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
   sequencer data into survival probabilities, certified fidelity bounds, and an optional plot,
   and packages everything in a `BenchmarkResult` container.

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

# 2. Sequencer: run a sqrt(X) (rx, pi/2) sequence at several depths, saving results as we go
sequencer = BenchmarkSequencer(backend=backend, qc=QuantumCircuit(1))
data = sequencer.run_circuits(
    depths=[0, 2, 4, 8, 16, 32],
    shots=2000,
    gate="rx",
    theta=np.pi/2,
    save=True,
    savename="x_gate_run",
)

# 3. Post-processing: turn counts into epsilon values, fidelity bounds and a plot
result = QSQPostProcessing(sequencer_data=sequencer.data)
print(result)
print("ε map:", result.epsilon_map)
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
| `measurement_basis` | `"X"`, `"Y"`, `"Z"` |
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
from qsqlab.data_processing import PostProcessing

# From data already in memory:
result = PostProcessing(sequencer_data=sequencer.data, show_plot=True)

# ...or from a previously saved run:
result = PostProcessing(
  sequencer_data="data/x_sequence.jsonl", 
  save=True, # optional save of created figure
  savefolder="figures" # folder path for figure-saving
  )

# ...or from the most recently modified file in the default data directory:
result = PostProcessing()

result.results          # BenchmarkResult(backend=..., gate=..., simulation=..., fidelity=..., depths=...)
result.points    # pandas.DataFrame, aggregated counts one entry per depth
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
- **Plotting** (`save=True`) renders and saves protocol-specific quantity vs. circuit
  length with error bars, styled via `DEFAULT_PLOT_STYLE` (overridable per call), and can
  save both `.png` and `.pdf` copies of the figure.
- **Results** renders relevant parameters and results into a `BenchmarkResult` container which can be either displayed with the markdown renderer of Jupyter notebooks, or printed out in the terminal.
- **Subclassing**: `PostProcessing` implements the shared plumbing (reading data, plotting,
  saving); protocol-specific logic belongs in a benchmark-specific subclass such as
  `QSQPostProcessing`. A subclass overrides `_validate_records` (what data it accepts),
  optionally `_process_points` (how raw records are aggregated), `_plot` (its figure) and
  `_create_results` (its `BenchmarkResult`); the parent's `_finalize` then ties them together.

#### `BenchmarkResult`

`BenchmarkResult` is the generic container every `PostProcessing` subclass returns (as
`result.results`). It bundles three things:

| Field | Type | Contents |
|---|---|---|
| `parameters` | `dict[str, Any]` | How the benchmark was run: backend, gates, simulation flag, depths, and any protocol-specific settings |
| `results` | `dict[str, Any]` | The benchmark outputs: fit values, fidelity bounds, error tables, ... |
| `figure` | `matplotlib.figure.Figure \| None` | The protocol's plot (optional) |

**Required keys.** On construction the container is validated and raises a `ValueError` listing
anything missing. Every container must have the parameters `backend`, `gates`, `simulation`,
`depths` and `post_processing` (class attribute `REQUIRED_PARAMETERS`). `REQUIRED_RESULTS` is empty
by default; a subclass can override either tuple to enforce its own keys.

**Access.** Entries can be read as dictionary items or directly as attributes. Attribute lookup
falls back to `parameters` first, then `results`:

```python
res = result.results

res.parameters["backend"]   # explicit
res.backend                 # same thing, via the attribute fallback
res.set_result("my_value", 0.99)        # add or update a result entry
res.set_parameter("note", "run 3")      # add or update a parameter entry
```

**Display.** The container renders itself depending on where it is shown:

- **Jupyter**: displaying the object (or setting `show_results=True` in a `PostProcessing`
  subclass) renders an HTML report via `to_html()` — a title, the figure embedded as a PNG, and
  side-by-side *Results* and *Parameters* tables.
- **Terminal / scripts**: `print(res)` gives a plain-text listing of results followed by parameters.

Formatting rules of the HTML report:

- **Math**: any string (key, value or title) containing `$...$` spans is rendered with matplotlib
  mathtext as inline SVG, e.g. `r"$\varepsilon_n$"`. A backslash-escaped `\$` is not treated as a
  delimiter. If an expression cannot be rendered, the raw text is shown instead.
- **Section headers**: a `results` or `parameters` key starting with `---` becomes a full-width
  section header labelled with the remaining text (its value is ignored, conventionally `None`). A
  bare `---` key produces a thin spacer.
- **Numbers**: floats are rounded to 5 digits inside lists and nested tables.
- **Long data collapses**: lists with more than 8 entries show a short preview that expands to the
  full list; dicts with more than 6 entries collapse into an expandable block. A dict of dicts is
  shown as a table (one row per outer key, one column per inner key).

> **Note:** `to_html()` closes the attached figure after embedding it, so the figure cannot be
> shown again after the report has been rendered. Save it first (e.g. with `save_plot=True`) if you
> need it afterwards.

The constants controlling this rendering (math colour, rounding digits, collapse thresholds and the
inline CSS) live in the `result_config` module, see [Configuration](#configuration).

### 4. `QSQPostProcessing`

`QSQPostProcessing` is the `PostProcessing` subclass that implements the single-qubit QSQ analysis found in [this paper by Nikolai Miklin et al.](https://doi.org/10.48550/arXiv.2608.17005)
From the measured zero-state probabilities `P[0|n]` of repeated `rx(π/2)` sequences at even depths
`n`, it derives the quantities `ε_n` and, from them, **certified (confidence-interval) bounds** on:

- the **short-sequence** gate fidelity lower bound (needs depths 0, 2 and 4), and
- the **phase**, **amplitude** and **long-sequence** fidelity lower bounds (needs user-chosen
  parameters `i`, `j` and optionally `ℓ`).

Everything is computed in the constructor; the results are exposed as properties and packaged in the
`BenchmarkResult` container.

```python
from qsqlab.data_processing import QSQPostProcessing

# From data already in memory (short-sequence analysis only):
result = QSQPostProcessing(sequencer_data=sequencer.data)

# ...or from a previously saved run, at 95% confidence, saving the plot:
result = QSQPostProcessing(
    sequencer_data="data/x_sequence.jsonl",
    precision=0.05,
    save_plot=True,
    savefolder="figures",
    savename="qsq_eps",
)

# ...or from the most recently modified file in the default data directory:
result = QSQPostProcessing()

result.results                # BenchmarkResult with all QSQ results and the epsilon plot
result.epsilon_map            # pandas.DataFrame, one entry per depth pair
result.short_seq_results      # {"high_fidelity_condition": Interval, "short_seq_fidelity": Interval}
```

#### Constructor arguments

| Argument | Default | Description |
|---|---|---|
| `sequencer_data` | `None` | Path to a `.jsonl` file or a `list[dict]`; `None` uses the most recent file in the default data folder |
| `precision` | `0.01` | Significance level `α` in `(0, 1)`. All intervals are reported at confidence `1 − precision`; *smaller* values mean higher confidence (and therefore wider intervals) |
| `long_seq_params` | `None` | `(i, j, ell)` for the long-sequence analysis, or `None` to skip it (see [below](#long-sequence-analysis)) |
| `show_results` | `True` | Display the result container with the Jupyter HTML renderer |
| `save_plot` | `False` | Save the epsilon plot as PDF and PNG |
| `savefolder` / `savename` | `""` | Folder and file name for the saved plot |
| `simulation` | `None` | Mark the data as simulated or measured; inferred from the backend when `None`. Controls the plot label and colour |
| `plot_style` | `None` | `matplotlib` rc overrides applied on top of `DEFAULT_PLOT_STYLE` |

#### Data requirements

`QSQPostProcessing` validates its input and raises a `ValueError` unless the data is genuine QSQ
data:

- The gate is `rx` with `θ = π/2` (use `gate="rx", theta=np.pi/2` in the sequencer).
- Every record was measured in the **Z basis**.
- **Only even depths are used.** Odd depths of the repeated `π/2` rotation have an ideal
  distribution of 0.5/0.5, which is an undefined ideal state, so they are dropped before analysis.
  `result.depths` still reports every depth present in the raw data, odd ones included.
- Each `ε_n` needs *both* `n` and `n + 2` to have been measured.

#### Epsilon values

For every even `n` where both `n` and `n + 2` were measured:

```math
ε_n = 1 − (−1)^{\frac{n}{2}} · ( \mathbb{P}[0|n] − \mathbb{P}[0|n+2] )
```

`result.epsilon_map` is a `pandas.DataFrame` indexed by depth `n` with the columns:

| Column | Meaning |
|---|---|
| `epsilon estimate` | Point estimate of `ε_n` |
| `confidence lower` / `confidence upper` | Bounds of the confidence interval |
| `uncertainty lower` / `uncertainty upper` | Distances from the estimate to the bounds, clipped at 0 (ready to use as asymmetric error bars) |

#### Statistical guarantees

All intervals are **Clopper–Pearson** intervals combined with a **Bonferroni correction**: the
allowed error `precision` is split across every probability `P[0|·]` that enters a given quantity, so
each reported interval holds with confidence of at least `1 − precision`. The scope differs per
quantity:

| Quantity | Coverage |
|---|---|
| Each `ε_n` in `epsilon_map` | Marginal: valid on its own, **not** jointly across different `n` |
| Short-sequence condition and fidelity | Joint (one box over `P[0|0]`, `P[0|2]`, `P[0|4]`) |
| Phase, `η_max` | Own box over the probabilities at `i, i+2j, i+4j, i+6j` (and their `+2` partners) |
| Amplitude | Own box (over `ℓ`, `ℓ+2` if `ℓ` is given) |
| Long-sequence fidelity | Joint (one box over *all* probabilities involved) |

Every interval is an `Interval(estimate, lower, upper)`. Quantities that cannot be certified are
filled with `NaN`, and in the compact result view they are shown as `not certified`.

#### Short-sequence analysis

Requires epsilon at depths 0 and 2, i.e. depths **0, 2 and 4** must all be measured. Otherwise a
warning is logged and `short_seq_results` is `None` (no exception is raised).

`result.short_seq_results` has two entries:

| Key | Contents |
|---|---|
| `"high_fidelity_condition"` | `Interval` for `√ε0 + √ε2` |
| `"short_seq_fidelity"` | `Interval` for the fidelity lower bound `F = 1/3 + 2/3 · √(1 − (√ε0 + √ε2)²)` |

Negative epsilon values are clipped to 0 before taking square roots. The fidelity bound only exists
where the square root is defined:

- If the **point estimate** of the condition exceeds 1, the whole fidelity interval is `NaN`.
- If only the **upper end** of the condition interval exceeds 1, the fidelity's lower bound is `NaN`
  (the estimate and upper bound are still reported).

> **The high-fidelity regime is not enforced.** The condition interval is returned so *you* can
> compare it to the threshold required for the bound to be valid (the source notes that the sum
> must stay below 0.5 for the intended confidence level).

#### Long-sequence analysis

The long-sequence analysis is optional and is requested with `long_seq_params=(i, j, ell)`:

| Parameter | Meaning |
|---|---|
| `i` | Starting depth (even) |
| `j` | Depth step (odd, positive). Epsilon must be available at depths `i`, `i+2j`, `i+4j` and `i+6j` |
| `ell` | Depth for the refined amplitude estimate (must be one of the `"ell"` candidates), or `None` to use the xi-based amplitude bound |

**Choosing parameters.** Valid choices are computed from your data and available as
`result.available_long_seq_parameters`:

```python
result = QSQPostProcessing(sequencer_data=data, show_results=False)
result.available_long_seq_parameters
# {"ij": [(i, j), ...], "ell": [n, ...]}
```

- `"ij"`: all `(i, j)` pairs for which epsilon is available at all four phase depths.
- `"ell"`: depths `n ≥ 22` with `|1 − ε_n| ≥ 0.7` at the point estimate. This is necessary but not
  sufficient: the analysis additionally requires the condition to hold over the whole confidence
  interval. An empty `"ell"` list still allows the analysis with `ell=None`.

A warning is logged if no `(i, j)` pair or no `ell` candidate exists, even if you did not request a
long-sequence analysis. Passing parameters the data does not support raises a `ValueError`. Once you
know your options, run the analysis:

```python
result = QSQPostProcessing(sequencer_data=data, long_seq_params=(2, 5, None))
result.long_seq_results["long seq fid"]     # Interval
```

**What is computed.** With the four epsilon values at `i, i+2j, i+4j, i+6j` sorted ascending as
`ε_0 ≤ ε_2j ≤ ε_4j ≤ ε_6j` (Lemmas 3–5 and Eq. 19 of the QSQ protocol):
```math
\textbf{Xi:}\;\;\;\;\;  
\xi=\sqrt{ \frac{\varepsilon_{6j} + \varepsilon_{4j} − \varepsilon_{2j} − \varepsilon_0}{1 − \varepsilon_{2j}} }
```
``` math
\textbf{Phase:}\;\;\;\;\; |\theta -\pi/2|\leq \frac{\arcsin(\xi)}{{2j}}
```
```math
\textbf{Tolerated initial deviation}\;|θ − π/2|: \;\;\;\;\; \eta_{max}= \frac{2\pi-\arcsin(\xi)}{{2j}}
```
```math
\textbf{Amplitude, "xi-based"   (l = None):}\;\;\;\;\;|\lambda| = (1 − \xi)^{1/(2j)}
```
```math
\textbf{Amplitude, "refined l-based" (l given)}:\;\;\;\;\;|\lambda|= |1 − \varepsilon_\ell|^{1/(\ell − 11)}
```
```math
\textbf{Long-sequence fidelity lower bound}:\;\;\;\;\; F_{long} = \frac{1}{3} + \frac{1}{3}(1+\sin{\theta})|\lambda|
```

The refined amplitude is certified only if `|1 − ε_ℓ| ≥ 0.7` holds over the whole interval.

**Results.** `result.long_seq_results` is a dict (or `None` if no parameters were given):

| Key | Contents |
|---|---|
| `"params"` | `{"i": ..., "j": ..., "ell": ...}` |
| `"phase"` / `"phase status"` | `Interval` for `θ` [rad], and `"ok"` or the reason it is not certified |
| `"eta max"` | `Interval` for `η_max` [rad]. Use its **lower** end as the initial trust required for the analysis: `|θ − π/2| ≤ η_max` |
| `"amplitude"` / `"amplitude status"` | `Interval` for `|λ|`, and `"ok"` or the reason it is not certified |
| `"amplitude method"` | `"xi-based"` or `"refined l-based"` |
| `"long seq fid"` / `"long seq fidelity status"` | `Interval` for the long-sequence fidelity lower bound, and `"ok"` or the first reason it is not certified |

#### Output: results container and plot

The `BenchmarkResult` in `result.results` is organized into three sections:

- **Short sequence results**: fidelity lower bound and high-fidelity condition in compact
  `estimate(error)` form, or a notice if depths 0, 2 and 4 were not available.
- **Long sequence results**: phase, `η_max`, amplitude and fidelity lower bound in compact form,
  plus `long sequence details` with every interval as `estimate [lower, upper]`. If no
  `long_seq_params` were given, a notice points you to the available options instead.
- **General results**: the available `(i, j)` pairs, the available `ℓ` depths, and an epsilon table
  with the estimate, lower uncertainty and lower confidence bound for each depth.

Its parameters contain the long-sequence parameters, backend, sorted gates, simulation flag, depths
and `"post_processing": "QSQ"`.

The attached figure plots `ε_n` against circuit length `n` with asymmetric error bars (marginal
intervals at `1 − precision`), on a fixed y-axis of `(0, 2.1)`. Points are **blue** for measured
data and **green** for simulated data. If long-sequence parameters were given, the depths `i`,
`i+2j`, `i+4j` and `i+6j` (and `ℓ`, when not `None`) are drawn on top in distinct colours, each with
its own legend entry. `save_plot=True` stores the figure as PDF and PNG.

#### Attributes and properties

| Property | Description |
|---|---|
| `precision` / `confidence` | Significance level `α` and confidence `1 − α` |
| `epsilon_map` | Epsilon table described [above](#epsilon-values) |
| `short_seq_results` | Short-sequence results, or `None` if depths 0, 2, 4 are not all available |
| `long_seq_parameters` | The `(i, j, ell)` given at construction, or `None` |
| `available_long_seq_parameters` | `{"ij": [...], "ell": [...]}` candidates found in the data |
| `long_seq_results` | Long-sequence results, or `None` |
| `results` | The `BenchmarkResult` container (inherited) |
| `points`, `data_raw`, `backend`, `gates`, `simulation`, `depths` | Inherited from `PostProcessing` |

#### Errors and warnings

| Raised / logged | When |
|---|---|
| `ValueError` | `precision` outside `(0, 1)`; `long_seq_params` not of length 3; data is not QSQ data; `long_seq_params` not supported by the data (missing epsilon depths, or `ell` not a candidate) |
| `NoAvailableEpsilon` | No even-depth records, or no depth pair `(n, n+2)` with even `n` was measured |
| Parent exceptions | Reading, validation, mixed backends (`MixedDataException`) and undefined ideal states propagate unchanged |
| Warning (logged) | Depths 0, 2, 4 not all available; no `(i, j)` pair or no `ell` candidate in the data |

`repr(result)` gives a one-line summary (precision, long-sequence parameters, backend, gates,
simulation flag, depth range and counts of depths and circuits).

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
| `result_config` | Display constants for the `BenchmarkResult` HTML report: math colour and regex, float rounding digits, list/dict collapse thresholds, and the inline CSS styles of the tables |

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