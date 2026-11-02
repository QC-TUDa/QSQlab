"""Tests for PostProcessing.
"""
import matplotlib
matplotlib.use("Agg")  # headless backend, no GUI windows during tests

from qsqlab.exceptions import EmptyDataFile, MixedBackendError, UndefinedIdealState
from qsqlab.data_processing.post_processing import PostProcessing, binomial_error
from qsqlab.data_processing import post_processing as pp_module
import pytest
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock


# --------------------------------------------------------------------------
# Fixtures & helpers
# --------------------------------------------------------------------------

def make_record(
    *,
    amount_of_gates=0,
    success_count=886,
    fail_count=114,
    ideal_success_label="0",
    backend="NoisyAerSimulator",
    gate="rx",
    shots=1000,
):
    """Build one sequencer record matching the documented jsonl schema."""
    fail_label = "1" if ideal_success_label == "0" else "0"
    return {
        "counts": {ideal_success_label: success_count, fail_label: fail_count},
        "ideal_probs": (
            {"0": 1.0, "1": 0.0} if ideal_success_label == "0" else {
                "0": 0.0, "1": 1.0}
        ),
        "gate": gate,
        "angles": {"theta": 1.5707963267948966, "phi": None, "lam": None},
        "amount_of_gates": amount_of_gates,
        "shots": shots,
        "measurement_basis": None,
        "channel_fid": 1.0,
        "backend": backend,
        "timestamp": "02-04-2026_16-22-39",
        "unix_time": 1775139759.355065,
    }


def write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture(autouse=True)
def patch_environment(monkeypatch):
    """Headless-ify plotting/display and stabilize SIMULATOR_BACKENDS/style for every test."""
    monkeypatch.setattr(pp_module, "display", MagicMock())
    monkeypatch.setattr(pp_module.plt, "show", MagicMock())
    monkeypatch.setattr(pp_module, "DEFAULT_PLOT_STYLE", {})
    monkeypatch.setattr(pp_module, "SIMULATOR_BACKENDS", {
                        "NoisyAerSimulator", "aer_simulator"})
    yield


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A tmp dir used as DEFAULT_DATA_ROOT for tests exercising the sequencer_data=None path."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(pp_module, "DEFAULT_DATA_ROOT", root)
    return root


@pytest.fixture
def two_depth_records():
    """Two depths, two records each, ideal state |0>, so 'success' == counts['0']."""
    return [
        make_record(amount_of_gates=0, success_count=980, fail_count=20),
        make_record(amount_of_gates=0, success_count=970, fail_count=30),
        make_record(amount_of_gates=10, success_count=800, fail_count=200),
        make_record(amount_of_gates=10, success_count=820, fail_count=180),
    ]


# --------------------------------------------------------------------------
# binomial_error
# --------------------------------------------------------------------------

class TestBinomialError:
    def test_matches_standard_formula(self):
        k, N = 80, 100
        p = k / N
        expected = np.sqrt(p * (1 - p) / N)
        assert binomial_error(k, N) == pytest.approx(expected)

    def test_zero_total_returns_zero_not_nan(self):
        # Regression test for the N == 0 guard.
        assert binomial_error(0, 0) == 0

    def test_all_successes_gives_zero_error(self):
        assert binomial_error(100, 100) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# _read
# --------------------------------------------------------------------------

@pytest.fixture
def pp():
    """Bare instance: skips __init__ so no reading/plotting/display happens."""
    return PostProcessing.__new__(PostProcessing)

class TestRead:
    def test_reads_valid_file_path_str(self, pp, tmp_path):
        path = write_jsonl(tmp_path / "run.jsonl",
                           [make_record(), make_record(amount_of_gates=5)])
        df = pp._read(str(path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_reads_valid_file_path_pathlib(self, pp, tmp_path):
        path = write_jsonl(tmp_path / "run.jsonl", [make_record()])
        df = pp._read(path)
        assert len(df) == 1

    def test_reads_list_of_dicts(self, pp):
        records = [make_record(), make_record(amount_of_gates=5)]
        df = pp._read(records)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_empty_list_raises_value_error(self, pp):
        with pytest.raises(ValueError):
            pp._read([])

    def test_list_missing_required_keys_raises_key_error(self, pp):
        # missing counts/amount_of_gates/angles/shots/ideal_probs
        bad_record = {"backend": "x", "gate": "rx"}
        with pytest.raises(KeyError):
            pp._read([bad_record])

    def test_invalid_type_raises_type_error(self, pp):
        with pytest.raises(TypeError):
            pp._read(12345)

    def test_none_with_missing_data_root_raises_file_not_found(self, pp, tmp_path, monkeypatch):
        missing_root = tmp_path / "does_not_exist"
        monkeypatch.setattr(pp_module, "DEFAULT_DATA_ROOT", missing_root)
        with pytest.raises(FileNotFoundError):
            pp._read(None)

    def test_none_with_no_jsonl_files_raises_file_not_found(self, pp, data_root):
        with pytest.raises(FileNotFoundError):
            pp._read(None)

    def test_none_reads_most_recently_modified_file(self, pp, data_root):
        older = write_jsonl(data_root / "older.jsonl",
                            [make_record(gate="rx")])
        newer = write_jsonl(data_root / "newer.jsonl",
                            [make_record(gate="ry")])
        now = time.time()
        os.utime(older, (now - 100, now - 100))
        os.utime(newer, (now, now))

        df = pp._read(None)
        assert set(df["gate"]) == {"ry"}

    def test_explicit_nonexistent_path_raises_file_not_found(self, pp, tmp_path):
        with pytest.raises(FileNotFoundError):
            pp._read(tmp_path / "nope.jsonl")

    def test_empty_file_raises_empty_data_file(self, pp, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(EmptyDataFile):
            pp._read(path)

    def test_blank_line_truncates_reading(self, pp, tmp_path):
        path = tmp_path / "run.jsonl"
        lines = [json.dumps(make_record()), "", json.dumps(
            make_record(amount_of_gates=99))]
        path.write_text("\n".join(lines))
        df = pp._read(path)
        assert len(df) == 1  # stopped at the blank line

    def test_malformed_json_line_truncates_reading(self, pp, tmp_path):
        path = tmp_path / "run.jsonl"
        lines = [json.dumps(make_record()), "{not valid json", json.dumps(
            make_record(amount_of_gates=99))]
        path.write_text("\n".join(lines))
        df = pp._read(path)
        assert len(df) == 1  # stopped at the malformed line

    def test_mixed_backends_in_file_raises(self, pp, tmp_path):
        path = write_jsonl(
            tmp_path / "run.jsonl",
            [make_record(backend="NoisyAerSimulator"),
             make_record(backend="real_hw")],
        )
        with pytest.raises(MixedBackendError):
            pp._read(path)

    def test_mixed_backends_in_list_raises(self, pp):
        records = [make_record(backend="NoisyAerSimulator"),
                   make_record(backend="real_hw")]
        with pytest.raises(MixedBackendError):
            pp._read(records)

    def test_file_record_missing_key_raises_key_error(self, pp, tmp_path):
        bad = make_record()
        del bad["counts"]
        path = write_jsonl(tmp_path / "run.jsonl", [make_record(), bad])
        with pytest.raises(KeyError, match="Record 1"):
            pp._read(path)


    def test_list_with_non_dict_record_raises_type_error(self, pp):
        with pytest.raises(TypeError):
            pp._read([make_record(), "not a dict"])


# --------------------------------------------------------------------------
# _process_points
# --------------------------------------------------------------------------

class TestProcessPoints:
    def test_aggregates_success_failure_by_depth(self, two_depth_records):
        df = pd.DataFrame(two_depth_records)
        points = PostProcessing._process_points(None, df)

        assert list(points.index) == [0, 10]
        assert points.loc[0, "successes"] == 980 + 970
        assert points.loc[0, "failures"] == 20 + 30
        assert points.loc[0, "total"] == 2000
        assert points.loc[10, "successes"] == 800 + 820
        assert points.loc[10, "total"] == 2000

    def test_success_prob_and_error_match_manual_calc(self, two_depth_records):
        df = pd.DataFrame(two_depth_records)
        points = PostProcessing._process_points(None, df)

        expected_prob_0 = (980 + 970) / 2000
        expected_err_0 = binomial_error(980 + 970, 2000)
        assert points.loc[0, "success_prob"] == pytest.approx(expected_prob_0)
        assert points.loc[0, "success_binomial_error"] == pytest.approx(
            expected_err_0)

    def test_ideal_state_one_flips_success_label(self):
        records = [
            make_record(amount_of_gates=0, ideal_success_label="1",
                        success_count=900, fail_count=100),
        ]
        df = pd.DataFrame(records)
        points = PostProcessing._process_points(None, df)
        # ideal is |1>, so counts['1'] (900) should count as success
        assert points.loc[0, "successes"] == 900
        assert points.loc[0, "failures"] == 100

    def test_undefined_ideal_state_raises(self):
        record = make_record()
        record["ideal_probs"] = {"0": 0.5, "1": 0.5}
        df = pd.DataFrame([record])
        with pytest.raises(UndefinedIdealState):
            PostProcessing._process_points(None, df)

    def test_does_not_mutate_input_dataframe(self, two_depth_records):
        df = pd.DataFrame(two_depth_records)
        original_columns = set(df.columns)
        PostProcessing._process_points(None, df)
        # no successes/failures/total leaked in
        assert set(df.columns) == original_columns

    def test_missing_ideal_probs_column_raises_key_error(self):
        record = make_record()
        del record["ideal_probs"]
        df = pd.DataFrame([record])
        with pytest.raises(KeyError):
            PostProcessing._process_points(None, df)


# --------------------------------------------------------------------------
# _determine_simulation
# --------------------------------------------------------------------------

class TestDetermineSimulation:
    def test_explicit_override_true(self):
        df = pd.DataFrame([make_record(backend="real_hw")])
        assert PostProcessing._determine_simulation(None, df, True) is True

    def test_explicit_override_false(self):
        df = pd.DataFrame([make_record(backend="NoisyAerSimulator")])
        assert PostProcessing._determine_simulation(None, df, False) is False

    def test_auto_detects_simulator_backend(self):
        df = pd.DataFrame([make_record(backend="NoisyAerSimulator")])
        assert PostProcessing._determine_simulation(None, df, None) is True

    def test_auto_detects_non_simulator_backend(self):
        df = pd.DataFrame([make_record(backend="real_hw")])
        assert PostProcessing._determine_simulation(None, df, None) is False

    def test_empty_data_raises_value_error(self):
        df = pd.DataFrame(columns=["backend"])
        with pytest.raises(ValueError):
            PostProcessing._determine_simulation(None, df, None)

    def test_mixed_backends_raises(self):
        df = pd.DataFrame(
            [make_record(backend="NoisyAerSimulator"),
             make_record(backend="real_hw")]
        )
        with pytest.raises(MixedBackendError):
            PostProcessing._determine_simulation(None, df, None)


# --------------------------------------------------------------------------
# _save_plot
# --------------------------------------------------------------------------

class TestSavePlot:
    def test_saves_png_and_pdf_with_given_name(self, tmp_path):
        fig, ax = pp_module.plt.subplots()
        png_path, pdf_path = PostProcessing._save_plot(
            None, fig, str(tmp_path), "myplot")
        assert png_path == tmp_path / "myplot.png"
        assert pdf_path == tmp_path / "myplot.pdf"
        assert png_path.exists()
        assert pdf_path.exists()

    def test_uses_timestamp_when_no_name_given(self, tmp_path):
        fig, ax = pp_module.plt.subplots()
        png_path, _ = PostProcessing._save_plot(None, fig, str(tmp_path), None)
        assert png_path.parent == tmp_path
        assert png_path.suffix == ".png"

    def test_creates_default_plots_dir_when_no_path_given(self, data_root):
        fig, ax = pp_module.plt.subplots()
        png_path, _ = PostProcessing._save_plot(None, fig, None, "x")
        assert png_path == data_root / "plots" / "x.png"
        assert png_path.exists()

    def test_closes_the_figure(self, tmp_path):
        fig, ax = pp_module.plt.subplots()
        PostProcessing._save_plot(None, fig, str(tmp_path), "closed")
        assert not pp_module.plt.fignum_exists(fig.number)


# --------------------------------------------------------------------------
# _plot (bare instance, only the attrs _plot actually reads)
# --------------------------------------------------------------------------

class TestPlot:
    def _bare_instance(self, data_raw, points, simulation):
        obj = PostProcessing.__new__(PostProcessing)
        obj._data_raw = data_raw
        obj._points = points
        obj._sim = simulation  # bypasses backend lookup entirely
        return obj

    def _points_df(self):
        return pd.DataFrame(
            {"success_prob": [0.99, 0.9],
                "success_binomial_error": [0.01, 0.02]},
            index=pd.Index([0, 10], name="amount_of_gates"),
        )

    def _data_raw_df(self):
        return pd.DataFrame(
            {"counts": {"0": 886, "1": 114}, "ideal_probs": {"0": 1.0, "1": 0.0}, "gate": "rx",
               "angles": {"theta": 1.5707963267948966, "phi": None, "lam": None}, 
               "amount_of_gates": 0, "shots": 1000, "measurement_basis": None, "channel_fid": 1.0, 
               "backend": "NoisyAerSimulator", "timestamp": "02-04-2026_16-22-39", 
               "unix_time": 1775139759.355065},
        )

    def test_axes_labels_and_ylim(self):
        obj = self._bare_instance(self._data_raw_df(), self._points_df(), simulation=False)
        fig, ax = obj._plot(save=False)
        assert ax.get_xlabel() == "Circuit Length $n$"
        assert ax.get_ylabel() == "Success Probability"
        assert ax.get_ylim() == (0, 1)

    def test_simulated_label_and_color(self):
        obj = self._bare_instance(self._data_raw_df(), self._points_df(), simulation=True)
        fig, ax = obj._plot(save=False)
        legend_text = ax.get_legend().get_texts()[0].get_text()
        assert legend_text == "Simulated data"
        line_color = ax.get_lines()[0].get_color()
        assert mcolors.same_color(line_color, "green")

    def test_measured_label_and_default_color(self):
        obj = self._bare_instance(self._data_raw_df(), self._points_df(), simulation=False)
        fig, ax = obj._plot(save=False)
        legend_text = ax.get_legend().get_texts()[0].get_text()
        assert legend_text == "Measured data"
        line_color = ax.get_lines()[0].get_color()
        assert not mcolors.same_color(line_color, "green")

    def test_save_true_writes_files_and_closes_figure(self, tmp_path):
        obj = self._bare_instance(self._data_raw_df(), self._points_df(), simulation=False)
        fig, ax = obj._plot(save=True,
                            savefolder=str(tmp_path), savename="saved")
        assert (tmp_path / "saved.png").exists()
        assert (tmp_path / "saved.pdf").exists()


# --------------------------------------------------------------------------
# Full construction & properties (integration-style, via __init__)
# --------------------------------------------------------------------------

class TestConstructionAndProperties:
    def test_full_construction_from_file(self, tmp_path, two_depth_records):
        path = write_jsonl(tmp_path / "run.jsonl", two_depth_records)
        instance = PostProcessing(sequencer_data=path)

        assert instance.backend == "NoisyAerSimulator"
        assert instance.gates == {"rx"}
        assert instance.depths == [0, 10]
        # NoisyAerSimulator is in patched SIMULATOR_BACKENDS
        assert instance.simulation is True
        assert len(instance.data_raw) == 4
        assert list(instance.points.index) == [0, 10]

    def test_display_called_on_init(self, tmp_path, two_depth_records):
        path = write_jsonl(tmp_path / "run.jsonl", two_depth_records)
        PostProcessing(sequencer_data=path)
        pp_module.display.assert_called_once()

    def test_simulation_override_takes_precedence(self, tmp_path, two_depth_records):
        path = write_jsonl(tmp_path / "run.jsonl", two_depth_records)
        instance = PostProcessing(sequencer_data=path, simulation=False)
        assert instance.simulation is False

    def test_multiple_gates_captured(self, tmp_path):
        records = [
            make_record(amount_of_gates=0, gate="rx"),
            make_record(amount_of_gates=0, gate="ry"),
        ]
        path = write_jsonl(tmp_path / "run.jsonl", records)
        instance = PostProcessing(sequencer_data=path)
        assert instance.gates == {"rx", "ry"}

    def test_construction_from_list_of_dicts(self, two_depth_records):
        instance = PostProcessing(sequencer_data=two_depth_records)
        assert len(instance.data_raw) == 4
        assert instance.depths == [0, 10]


# --------------------------------------------------------------------------
# results property
# --------------------------------------------------------------------------

class TestResultsProperty:
    def test_parameters_reflect_run(self, tmp_path, two_depth_records):
        path = write_jsonl(tmp_path / "run.jsonl", two_depth_records)
        instance = PostProcessing(sequencer_data=path)
        result = instance.results

        assert result.parameters["backend"] == "NoisyAerSimulator"
        assert result.parameters["gates"] == ["rx"]
        assert result.parameters["depths"] == [0, 10]
        assert result.parameters["post_processing"] == "Generic"

    def test_success_probabilities_present_for_each_depth(self, tmp_path, two_depth_records):
        path = write_jsonl(tmp_path / "run.jsonl", two_depth_records)
        instance = PostProcessing(sequencer_data=path)
        probs = instance.results.results["success_probabilities"]
        assert set(probs.keys()) == {0, 10}
        assert "success_prob" in probs[0]
        assert "success_binomial_error" in probs[0]


# --------------------------------------------------------------------------
# __repr__ / __str__
# --------------------------------------------------------------------------

class TestReprAndStr:
    def test_repr_contains_key_fields(self, tmp_path, two_depth_records):
        path = write_jsonl(tmp_path / "run.jsonl", two_depth_records)
        instance = PostProcessing(sequencer_data=path)
        r = repr(instance)
        assert "PostProcessing" in r
        assert "NoisyAerSimulator" in r
        assert "rx" in r
        assert "n_circuits=4" in r

    def test_str_delegates_to_results(self, tmp_path, two_depth_records):
        path = write_jsonl(tmp_path / "run.jsonl", two_depth_records)
        instance = PostProcessing(sequencer_data=path)
        assert str(instance) == str(instance.results)

    def test_repr_never_raises_on_broken_data(self):
        obj = PostProcessing.__new__(PostProcessing)
        obj._data_raw = pd.DataFrame(
            {"backend": [], "gate": [], "amount_of_gates": []})
        obj._sim = None
        r = repr(obj)  # must not raise
        assert r == "<PostProcessing (data unavailable)>"
