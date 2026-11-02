import re

# Matches ``$...$`` math spans; a backslash-escaped ``\$`` is not a delimiter.
_MATH_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")

_MATH_COLOR = "#5b9bd5"
_FLOAT_DIGITS = 5               # floats are rounded to this many digits in tables
_LIST_COLLAPSE_THRESHOLD = 8    # lists longer than this are collapsed
_DICT_COLLAPSE_THRESHOLD = 6    # dicts with more entries than this are collapsed

# Inline CSS shared by the HTML tables.
_NESTED_TABLE_STYLE = (
    "margin-top:4px; border-collapse:collapse; "
    "font-family:monospace; font-size:12px;"
)
_NESTED_TH_STYLE = "padding:2px 8px; text-align:left; border-bottom:1px solid #ccc;"
_NESTED_TD_STYLE = "padding:2px 8px; border-bottom:1px solid #eee;"
_NESTED_KEY_STYLE = _NESTED_TD_STYLE + " text-align:left;"
_NESTED_VALUE_STYLE = _NESTED_KEY_STYLE + " white-space:nowrap;"
_ROW_CELL_STYLE = "padding:4px 12px; border-bottom:1px solid #eee; text-align:left;"
_SECTION_LABEL_STYLE = (
    "padding:6px 12px; background:rgba(128,128,128,0.12); "
    "border-top:1px solid #ccc; border-bottom:1px solid #ccc; "
    "text-align:center; font-weight:bold; font-size:12px;"
)
_SECTION_SPACER_STYLE = "padding:0; height:6px; border-bottom:2px solid #ccc;"
_FIGURE_SEPARATOR = (
    '<hr style="border:none; border-top:1px solid #ccc; margin:16px 0 16px 0;">'
)
