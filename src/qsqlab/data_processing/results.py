"""Result container for quantum computer benchmarks.

:class:`BenchmarkResult` bundles the parameters, results and (optionally) the
figure of a benchmark run. It can render itself as a self-contained HTML
report (used by Jupyter via ``_repr_html_``) and as plain text (``str``).

Any string value, key or title containing ``$...$`` spans is rendered with
matplotlib's mathtext and embedded as an inline SVG, so LaTeX-style labels
such as ``r"$\\varepsilon_n$"`` display properly in the HTML report.
"""

import base64
import html
import io
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..config.result_config import (
    _MATH_RE, 
    _MATH_COLOR, 
    _FLOAT_DIGITS, 
    _LIST_COLLAPSE_THRESHOLD, 
    _DICT_COLLAPSE_THRESHOLD, 
    _NESTED_TABLE_STYLE, 
    _NESTED_TH_STYLE, 
    _NESTED_TD_STYLE, 
    _NESTED_KEY_STYLE,
    _NESTED_VALUE_STYLE, 
    _ROW_CELL_STYLE, 
    _SECTION_LABEL_STYLE, 
    _SECTION_SPACER_STYLE, 
    _FIGURE_SEPARATOR
)

# --------------------------------------------------------------------------
# Math rendering
# --------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _math_to_img(expr: str, fontsize: int = 12) -> str:
    """Render a mathtext expression to an inline SVG.

    Parameters
    ----------
    expr : str
        Expression *without* the surrounding ``$`` delimiters.
    fontsize : int, optional
        Font size of the rendered expression, by default ``12``.

    Returns
    -------
    str
        An ``<svg>`` element, or the HTML-escaped original ``$expr$`` text if
        rendering fails (e.g. invalid mathtext).
    """
    try:
        # Figure() is used directly (not plt.figure) so nothing is registered
        # with pyplot's global figure manager.
        fig = Figure()
        fig.text(0, 0, f"${expr}$", fontsize=fontsize, color=_MATH_COLOR)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight",
                    pad_inches=0.02, transparent=True)
        svg = buf.getvalue().decode("utf-8")
        svg = svg[svg.index("<svg"):]  # drop XML header / doctype
        return svg.replace("<svg ", '<svg style="vertical-align:middle;" ', 1)
    except Exception:
        return html.escape(f"${expr}$")


def _tex(value: Any) -> Any:
    """HTML-escape plain text and render ``$...$`` spans as math images.

    Parameters
    ----------
    value : Any
        Value to convert. Non-string values are returned unchanged.

    Returns
    -------
    Any
        HTML string for string input, otherwise ``value`` itself.
    """
    if not isinstance(value, str):
        return value

    parts, last = [], 0
    for match in _MATH_RE.finditer(value):
        parts.append(html.escape(value[last:match.start()]))
        parts.append(_math_to_img(match.group(1)))
        last = match.end()
    parts.append(html.escape(value[last:]))
    return "".join(parts)


# --------------------------------------------------------------------------
# HTML formatting helpers
# --------------------------------------------------------------------------

def _round_cell(value: Any) -> Any:
    """Round floats for display; leave everything else untouched."""
    return round(value, _FLOAT_DIGITS) if isinstance(value, float) else value


def _collapsible(summary: str, body: str) -> str:
    """Wrap ``body`` in a ``<details>`` element with the given ``summary``."""
    return (
        f'<details>'
        f'<summary style="cursor:pointer;">{summary}</summary>'
        f'{body}'
        f'</details>'
    )


def _format_list(values: list) -> str:
    """Format a list as ``[a, b, ...]``, collapsing long lists.

    Lists longer than ``_LIST_COLLAPSE_THRESHOLD`` show a preview (first two
    and last element, rounded) and expand to the full, unrounded list.
    """
    def item(x: Any) -> str:
        return html.escape(str(_round_cell(x)))

    if len(values) <= _LIST_COLLAPSE_THRESHOLD:
        return f"[{', '.join(item(x) for x in values)}]"

    preview = f"[{item(values[0])}, {item(values[1])}, ..., {item(values[-1])}]"
    full = f"[{', '.join(html.escape(str(x)) for x in values)}]"
    full_div = (
        f'<div style="margin-top:4px; font-family:monospace; font-size:12px; '
        f'max-width:300px; word-wrap:break-word;">{full}</div>'
    )
    return _collapsible(f"{preview} ({len(values)} values)", full_div)


def _format_dict_of_dicts(data: dict) -> str:
    """Format ``{outer: {inner: value}}`` as a table.

    One row per outer key and one column per inner key. Columns are the union
    of all inner keys in first-seen order; missing entries are left blank.
    """
    columns: list = []
    for sub in data.values():
        for key in sub:
            if key not in columns:
                columns.append(key)

    header_cells = "".join(
        f'<th style="{_NESTED_TH_STYLE}">{_tex(col)}</th>' for col in columns
    )
    header = (
        f'<tr><th style="{_NESTED_TH_STYLE}">key</th>{header_cells}</tr>'
    )
    body = "".join(
        f'<tr><td style="{_NESTED_TD_STYLE}">{_tex(outer_key)}</td>'
        + "".join(
            f'<td style="{_NESTED_TD_STYLE}">'
            f'{_tex(_round_cell(sub[col])) if col in sub else ""}</td>'
            for col in columns
        )
        + '</tr>'
        for outer_key, sub in data.items()
    )
    return f'<table style="{_NESTED_TABLE_STYLE}">{header}{body}</table>'


def _format_flat_dict(data: dict) -> str:
    """Format a flat or mixed dict as key/value rows (values formatted recursively)."""
    rows = "".join(
        f'<tr><td style="{_NESTED_KEY_STYLE}">{_tex(key)}</td>'
        f'<td style="{_NESTED_VALUE_STYLE}">{_format_value(val)}</td></tr>'
        for key, val in data.items()
    )
    return f'<table style="{_NESTED_TABLE_STYLE}">{rows}</table>'


def _format_dict(data: dict) -> str:
    """Format a dict as a nested table, collapsing large ones."""
    if not data:
        return ""

    if all(isinstance(sub, dict) for sub in data.values()):
        inner = _format_dict_of_dicts(data)
    else:
        inner = _format_flat_dict(data)

    if len(data) > _DICT_COLLAPSE_THRESHOLD:
        return _collapsible(f"{len(data)} values", inner)
    return inner


def _format_value(value: Any) -> Any:
    """Format a table value: lists and dicts get structure, the rest ``_tex``."""
    if isinstance(value, list):
        return _format_list(value)
    if isinstance(value, dict):
        return _format_dict(value)
    return _tex(value)


def _table_row(key: Any, value: Any) -> str:
    """Build one row of a two-column table.

    A string key starting with ``---`` produces a full-width section header
    labelled with the remaining text, or a thin spacer if the label is empty.
    """
    if isinstance(key, str) and key.startswith("---"):
        label = key.lstrip("-").strip()
        if label:
            return (
                f'<tr><td colspan="2" style="{_SECTION_LABEL_STYLE}">'
                f'{_tex(label)}</td></tr>'
            )
        return f'<tr><td colspan="2" style="{_SECTION_SPACER_STYLE}"></td></tr>'

    return (
        f'<tr><td style="{_ROW_CELL_STYLE}">{_tex(key)}</td>'
        f'<td style="{_ROW_CELL_STYLE}">{_format_value(value)}</td></tr>'
    )


def _table(title: str, data: dict) -> str:
    """Build a titled card containing a key/value table of ``data``."""
    rows = "".join(_table_row(k, v) for k, v in data.items())
    return (
        '<div style="border:1px solid #ddd; border-radius:8px; '
        'overflow:hidden; min-width:220px;">'
        '<div style="font-size:14px; color:#000000; background:#f5f5f5; '
        'padding:6px 0px; font-weight:bold; text-align:center; '
        f'border-bottom:1px solid #ddd;">{title}</div>'
        f'<table style="border-collapse:collapse; width:100%;">{rows}</table>'
        '</div>'
    )


def _figure_to_html(fig: Figure, width: int = 400) -> str:
    """Embed a figure as a base64 PNG ``<img>`` followed by a separator.

    Note
    ----
    The figure is closed after rendering, as it was before this refactor.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return (
        f'<div style="text-align:center;">'
        f'<img src="data:image/png;base64,{data}" width="{width}"></div>'
        f'{_FIGURE_SEPARATOR}'
    )


# --------------------------------------------------------------------------
# Public container
# --------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Generic result container for quantum computer benchmarks.

    Parameters and results are also reachable as attributes, e.g.
    ``result.backend`` falls back to ``result.parameters["backend"]``
    (parameters take precedence over results on name clashes).

    Parameters
    ----------
    parameters : dict[str, Any]
        Metadata describing how the benchmark was run (backend, gates,
        simulation flag, depths, and any benchmark-specific settings).
        Must contain every key in :attr:`REQUIRED_PARAMETERS`.
    results : dict[str, Any]
        Benchmark outputs (fit values, fidelity results, errors, etc.).
        Must contain every key in :attr:`REQUIRED_RESULTS`. A string key
        starting with ``---`` is rendered as a section header in the HTML
        report (its value is ignored, conventionally ``None``).
    figure : matplotlib.figure.Figure | None, optional
        Figure with benchmark-specific data points, by default ``None``.

    Attributes
    ----------
    REQUIRED_PARAMETERS : ClassVar[tuple[str, ...]]
        Parameter keys that must be present.
    REQUIRED_RESULTS : ClassVar[tuple[str, ...]]
        Result keys that must be present. Subclasses may override.

    Raises
    ------
    ValueError
        On construction, if any required parameter or result key is missing.
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    figure: Figure | None = None

    REQUIRED_PARAMETERS: ClassVar[tuple[str, ...]] = (
        "backend", "gates", "simulation", "depths", "post_processing",
    )
    REQUIRED_RESULTS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that all required parameter and result keys are present.

        Raises
        ------
        ValueError
            Listing the missing parameter and result keys.
        """
        missing_p = [
            k for k in self.REQUIRED_PARAMETERS if k not in self.parameters]
        missing_r = [k for k in self.REQUIRED_RESULTS if k not in self.results]
        if missing_p or missing_r:
            raise ValueError(
                f"Missing required parameters={missing_p}, results={missing_r}"
            )

    def set_result(self, key: str, value: Any) -> None:
        """Add or update a result entry."""
        self.results[key] = value

    def set_parameter(self, key: str, value: Any) -> None:
        """Add or update a parameter entry."""
        self.parameters[key] = value

    def to_html(self) -> str:
        """Render the container as an HTML report.

        The report has a title (``parameters["post_processing"]``, falling
        back to the class name), the figure if present, and side-by-side
        "Results" and "Parameters" tables. Floats are rounded to 5 digits,
        long lists and large dicts are collapsible, and ``$...$`` spans are
        rendered as math.

        Returns
        -------
        str
            Self-contained HTML (figure embedded as base64 PNG).

        Note
        ----
        If a figure is present it is closed after rendering.
        """
        title = _tex(self.parameters.get(
            "post_processing", self.__class__.__name__))
        figure_html = (
            _figure_to_html(
                self.figure, width=640) if self.figure is not None else ""
        )
        return (
            '<div style="text-align:center; font-size:22px; font-weight:bold; '
            f'margin-bottom:4px;">{title} Results</div>\n'
            '<hr style="border:none; border-top:1px solid #ccc; margin:0 0 16px 0;">\n'
            f'{figure_html}\n'
            '<div style="display:flex; gap:16px; justify-content:center; margin-top:16px;">'
            f'<div>{_table("Results", self.results)}</div>'
            f'<div>{_table("Parameters", self.parameters)}</div>'
            '</div>'
        )

    def _repr_html_(self) -> str:
        """Rich display hook for Jupyter; see :meth:`to_html`."""
        return self.to_html()

    def __str__(self) -> str:
        """Plain-text summary: results first, then parameters.

        This is the display format when running post-processing classes
        from a ``.py`` script.
        """
        lines = [f"{self.__class__.__name__}("]
        lines += [f"  {k} = {v}," for k, v in self.results.items()]
        lines += [f"  {k} = {v}," for k, v in self.parameters.items()]
        lines.append(")")
        return "\n".join(lines)

    def __getattr__(self, name: str) -> Any:
        """Fall back to ``parameters`` then ``results`` for attribute access.

        Only called when normal attribute lookup fails. The two container
        names are excluded explicitly to avoid infinite recursion when they
        are not yet set (e.g. during copying or unpickling).

        Raises
        ------
        AttributeError
            If ``name`` is in neither dict.
        """
        if name in ("parameters", "results"):
            raise AttributeError(name)
        if name in self.parameters:
            return self.parameters[name]
        if name in self.results:
            return self.results[name]
        raise AttributeError(
            f"{self.__class__.__name__!r} has no attribute {name!r}")


# import html
# import re
# import base64
# import io

# from functools import lru_cache
# from matplotlib.figure import Figure
# from dataclasses import dataclass, field
# from typing import Any, ClassVar

# import matplotlib.pyplot as plt

# _MATH_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")

# @dataclass
# class BenchmarkResult:
#     """
#     Generic result container for quantum computer benchmarks.

#     Parameters
#     ----------
#     parameters : dict[str, Any]
#         Metadata describing how the benchmark was run (backend, gates,
#         simulation flag, depths, and any benchmark-specific settings).
#     results : dict[str, Any]
#         Benchmark outputs (fit values, fidelity results, errors, etc.).
#     figure : matplotlib.pyplot.Figure | None
#         matplotlib figure with Benchmark-specific data points.
#         by default None.
#     """
#     parameters: dict[str, Any] = field(default_factory=dict)
#     results: dict[str, Any] = field(default_factory=dict)
#     figure: plt.Figure | None = None

#     REQUIRED_PARAMETERS: ClassVar[tuple[str]] = (
#         "backend", "gates", "simulation", "depths", "post_processing",
#         )
#     REQUIRED_RESULTS: ClassVar[tuple[str]] = ()

#     def set_result(self, key: str, value: Any) -> None:
#         """Add or update a result entry."""
#         self.results[key] = value

#     def set_parameter(self, key: str, value: Any) -> None:
#         """Add or update a parameter entry."""
#         self.parameters[key] = value

#     def to_html(self) -> str:
#         import textwrap

#         def fmt_cell(v):
#             """Format a single scalar-ish value (round floats, leave the rest)."""
#             if isinstance(v, float):
#                 return round(v, 5)
#             return v


#         def table(title, d):
#             def fmt(v):
#                 if isinstance(v, list):
#                     def item(x):
#                         return html.escape(str(fmt_cell(x)))

#                     if len(v) > 8:
#                         preview = f"[{item(v[0])}, {item(v[1])}, ..., {item(v[-1])}]"
#                         full = f"[{', '.join(html.escape(str(x)) for x in v)}]"
#                         return (
#                             f'<details>'
#                             f'<summary style="cursor:pointer;">{preview} ({len(v)} values)</summary>'
#                             f'<div style="margin-top:4px; font-family:monospace; font-size:12px; '
#                             f'max-width:300px; word-wrap:break-word;">{full}</div>'
#                             f'</details>'
#                         )
#                     return f"[{', '.join(item(x) for x in v)}]"
    
#                 if isinstance(v, dict):
#                     if not v:
#                         return ""

#                     if all(isinstance(sub, dict) for sub in v.values()):
#                         # dict of dicts -> one row per outer key, one column per inner key
#                         columns = []
#                         for sub in v.values():
#                             for key in sub:
#                                 if key not in columns:
#                                     columns.append(key)

#                         header_cells = "".join(
#                             f'<th style="padding:2px 8px; text-align:left; border-bottom:1px solid #ccc;">{_tex(col)}</th>'
#                             for col in columns
#                         )
#                         header = (
#                             f'<tr>'
#                             f'<th style="padding:2px 8px; text-align:left; border-bottom:1px solid #ccc;">key</th>'
#                             f'{header_cells}'
#                             f'</tr>'
#                         )
#                         body_rows = "".join(
#                             f'<tr>'
#                             f'<td style="padding:2px 8px; border-bottom:1px solid #eee;">{_tex(outer_key)}</td>'
#                             + "".join(
#                                 f'<td style="padding:2px 8px; border-bottom:1px solid #eee;">'
#                                 f'{_tex(fmt_cell(sub[col])) if col in sub else ""}</td>'
#                                 for col in columns
#                             )
#                             + f'</tr>'
#                             for outer_key, sub in v.items()
#                         )
#                         inner = (
#                             f'<table style="margin-top:4px; border-collapse:collapse; '
#                             f'font-family:monospace; font-size:12px;">{header}{body_rows}</table>'
#                         )
#                     else:
#                         # flat or mixed dict -> key/value rows; fmt() recurses into nested values
#                         kv_rows = "".join(
#                             f'<tr>'
#                             f'<td style="padding:2px 8px; border-bottom:1px solid #eee; text-align:left;">{_tex(key)}</td>'
#                             f'<td style="padding:2px 8px; border-bottom:1px solid #eee; text-align:left; '
#                             f'white-space:nowrap;">{fmt(val)}</td>'
#                             f'</tr>'
#                             for key, val in v.items()
#                         )
#                         inner = (
#                             f'<table style="margin-top:4px; border-collapse:collapse; '
#                             f'font-family:monospace; font-size:12px;">{kv_rows}</table>'
#                         )

#                     if len(v) > 6:
#                         return (
#                             f'<details>'
#                             f'<summary style="cursor:pointer;">{len(v)} values</summary>'
#                             f'{inner}'
#                             f'</details>'
#                         )
#                     return inner

#                 return _tex(v)

#             def row(k, v):
#                 # separator row: key starts with "---"
#                 if isinstance(k, str) and k.startswith("---"):
#                     label = k.lstrip("-").strip()
#                     if label:
#                         return (
#                             '<tr><td colspan="2" style="padding:6px 12px; '
#                             'background:rgba(128,128,128,0.12); '
#                             'border-top:1px solid #ccc; border-bottom:1px solid #ccc; '
#                             f'text-align:center; font-weight:bold; font-size:12px;">{_tex(label)}</td></tr>'
#                         )
#                     return (
#                         '<tr><td colspan="2" style="padding:0; height:6px; '
#                         'border-bottom:2px solid #ccc;"></td></tr>'
#                     )
#                 return (
#                     f'<tr><td style="padding:4px 12px; border-bottom:1px solid #eee; text-align:left;">{_tex(k)}</td>'
#                     f'<td style="padding:4px 12px; border-bottom:1px solid #eee; text-align:left;">{fmt(v)}</td></tr>'
#                 )

#             rows = "".join(row(k, v) for k, v in d.items())
#             return f"""
#             <div style="border:1px solid #ddd; border-radius:8px; overflow:hidden; min-width:220px;">
#             <div style="font-size:14px; color: #000000; background:#f5f5f5; padding:6px 0px; font-weight:bold; text-align:center; border-bottom:1px solid #ddd;">{title}</div>
#             <table style="border-collapse:collapse; width:100%;">{rows}</table>
#             </div>
#             """

#         def fig_to_base64_html(fig: plt.Figure, width: int = 400) -> str:
#             buf = io.BytesIO()
#             fig.savefig(buf, format="png", bbox_inches="tight")
#             buf.seek(0)
#             data = base64.b64encode(buf.read()).decode("utf-8")
#             plt.close(fig)
#             return f'<div style="text-align:center;"><img src="data:image/png;base64,{data}" width="{width}">\
#                 </div><hr style="border:none; border-top:1px solid #ccc; margin:16px 0 16px 0;">'

#         page = f"""
#         <div style="text-align:center; font-size:22px; font-weight:bold; margin-bottom:4px;">
#         {_tex(self.parameters.get("post_processing", self.__class__.__name__))} Results
#         </div>
#         <hr style="border:none; border-top:1px solid #ccc; margin:0 0 16px 0;">

#         {fig_to_base64_html(self.figure, width=640) if self.figure is not None else ""}

#         <div style="display:flex; gap:16px; justify-content:center; margin-top:16px;">
#         <div>{table("Results", self.results)}</div>
#         <div>{table("Parameters", self.parameters)}</div>
#         </div>
#         """
#         return textwrap.dedent(page).strip()

#     def _validate(self) -> None:
#         missing_p = [k for k in self.REQUIRED_PARAMETERS if k not in self.parameters]
#         missing_r = [k for k in self.REQUIRED_RESULTS if k not in self.results]
#         if missing_p or missing_r:
#             raise ValueError(
#                 f"Missing required parameters={missing_p}, results={missing_r}"
#             )
    
#     def _repr_html_(self) -> str:
#         return self.to_html()

#     def __post_init__(self) -> None:
#         self._validate()

#     def __str__(self) -> str: # Intended display format when executing PostProcessing classes in .py
#         lines = [f"{self.__class__.__name__}("]
#         lines += [f"  {k} = {v}," for k, v in self.results.items()]
#         lines += [f"  {k} = {v}," for k, v in self.parameters.items()]
#         lines.append(")")
#         return "\n".join(lines)

#     def __getattr__(self, name: str) -> Any:
#         # only called when normal attribute lookup fails
#         if name in ("parameters", "results"):
#             raise AttributeError(name)
#         if name in self.parameters:
#             return self.parameters[name]
#         if name in self.results:
#             return self.results[name]
#         raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")


# @lru_cache(maxsize=256)
# def _math_to_img(expr: str, fontsize: int = 12,) -> str:
#     """Render a LaTeX expression (without the $ delimiters) to an inline SVG."""
#     try:
#         fig = Figure()
#         fig.text(0, 0, f"${expr}$", fontsize=fontsize, color="#5b9bd5")  # "#8a8a8a")
#         buf = io.BytesIO()
#         fig.savefig(buf, format="svg", bbox_inches="tight",
#                     pad_inches=0.02, transparent=True)
#         svg = buf.getvalue().decode("utf-8")
#         svg = svg[svg.index("<svg"):]   # drop XML header / doctype
#         svg = svg.replace("<svg ", '<svg style="vertical-align:middle;" ', 1)
#         return svg
#     except Exception:
#         return html.escape(f"${expr}$")


# def _tex(s: Any) -> Any:
#     """HTML-escape plain text and render any $...$ spans as math images."""
#     if not isinstance(s, str):
#         return s
#     out, last = [], 0
#     for m in _MATH_RE.finditer(s):
#         out.append(html.escape(s[last:m.start()]))
#         out.append(_math_to_img(m.group(1)))
#         last = m.end()
#     out.append(html.escape(s[last:]))
#     return "".join(out)
