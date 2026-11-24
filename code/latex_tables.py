"""Publication tables in the style of arXiv:2510.01644v2.

Table 4 (known jailbreaks) has a feature column, a model column, and mean/std
pairs for AUC, Accuracy, FNR, and TPR. Table 5 (novel jailbreaks) has one row
per held-out type. The jailbreak-type table is the journal extension: one row
per type with Accuracy, F1, AUC, TPR, and FNR.

Numbers use three decimal places so a standard deviation of 0.004 is not
rounded away. The population standard deviation matches ``numpy.std`` (the
notebooks' aggregator). Missing scores render as ``--``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from metrics import is_missing_number
from paths import results_dir

BINARY_CAPTION = (
    "Performance on known jailbreak patterns. "
    "Each experiment run uses a randomly selected training/test split. "
    "Entries are the mean and standard deviation across runs."
)
NOVEL_CAPTION = (
    "Performance on novel jailbreak patterns. "
    "Each row is a jailbreak type held out of training and used only for testing."
)
CATEGORY_CAPTION = (
    "One-vs-rest jailbreak type classification. "
    "F1, TPR, and FNR treat the named jailbreak type as the positive class."
)

BINARY_METRICS = ("auc", "accuracy", "fnr", "tpr")
BINARY_HEADERS = ("AUC", "Accuracy", "FNR", "TPR")


def population_std(values) -> float:
    numbers = [float(value) for value in values]
    count = len(numbers)
    if count == 0:
        return float("nan")
    mean = sum(numbers) / count
    variance = sum((value - mean) ** 2 for value in numbers) / count
    return math.sqrt(variance)


def summarize_binary_runs(runs: list[dict]) -> list[dict]:
    """Aggregate per-run binary metrics, preserving first-seen model order."""
    if not runs:
        raise ValueError("No binary runs to summarise")
    groups: dict[tuple[str, str], list[dict]] = {}
    order: list[tuple[str, str]] = []
    for run in runs:
        _require(run, ("features", "model", *BINARY_METRICS))
        key = (str(run["features"]), str(run["model"]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(run)
    summary = []
    for features, model in order:
        items = groups[(features, model)]
        row = {"features": features, "model": model, "n_runs": len(items)}
        for metric in BINARY_METRICS:
            values = [_as_float(item[metric], metric) for item in items]
            row[f"{metric}_mean"] = sum(values) / len(values)
            row[f"{metric}_std"] = population_std(values)
        summary.append(row)
    return summary


def binary_detection_table(
    runs: list[dict],
    caption: str = BINARY_CAPTION,
    label: str = "tab:binary-detection",
    decimals: int = 3,
) -> str:
    summary = summarize_binary_runs(runs)
    header = [
        "Features",
        "Model",
        r"\multicolumn{2}{c}{AUC}",
        r"\multicolumn{2}{c}{Accuracy}",
        r"\multicolumn{2}{c}{FNR}",
        r"\multicolumn{2}{c}{TPR}",
    ]
    subhead = ["", "", "Mean", "Std", "Mean", "Std", "Mean", "Std", "Mean", "Std"]
    body = []
    for row in summary:
        cells = [_escape(row["features"]), _escape(row["model"])]
        for metric in BINARY_METRICS:
            cells.append(_format_number(row[f"{metric}_mean"], decimals))
            cells.append(_format_number(row[f"{metric}_std"], decimals))
        body.append(cells)
    return _table(
        column_spec="llrrrrrrrr",
        caption=caption,
        label=label,
        header_rows=[header, subhead],
        body_rows=body,
        cmidrules=(
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}"
        ),
    )


def novel_jailbreak_table(
    rows: list[dict],
    caption: str = NOVEL_CAPTION,
    label: str = "tab:novel-jailbreak",
    decimals: int = 3,
) -> str:
    if not rows:
        raise ValueError("No novel-jailbreak rows to tabulate")
    body = []
    for row in rows:
        _require(row, ("held_out_type", "auc", "accuracy", "fnr", "tpr"))
        body.append(
            [
                _escape(row["held_out_type"]),
                _format_number(row["auc"], decimals),
                _format_number(row["accuracy"], decimals),
                _format_number(row["fnr"], decimals),
                _format_number(row["tpr"], decimals),
            ]
        )
    return _table(
        column_spec="lrrrr",
        caption=caption,
        label=label,
        header_rows=[["Held-out Jailbreak Type", "AUC", "Accuracy", "FNR", "TPR"]],
        body_rows=body,
    )


def jailbreak_type_table(
    rows: list[dict],
    caption: str = CATEGORY_CAPTION,
    label: str = "tab:jailbreak-type",
    decimals: int = 3,
) -> str:
    if not rows:
        raise ValueError("No jailbreak-type rows to tabulate")
    body = []
    for row in rows:
        _require(row, ("jailbreak_type", "accuracy", "f1", "auc", "tpr", "fnr"))
        body.append(
            [
                _escape(row["jailbreak_type"]),
                _format_number(row["accuracy"], decimals),
                _format_number(row["f1"], decimals),
                _format_number(row["auc"], decimals),
                _format_number(row["tpr"], decimals),
                _format_number(row["fnr"], decimals),
            ]
        )
    return _table(
        column_spec="lrrrrr",
        caption=caption,
        label=label,
        header_rows=[["Jailbreak Type", "Accuracy", "F1", "AUC", "TPR", "FNR"]],
        body_rows=body,
    )


def export_results(
    stem: str,
    kind: str,
    records: list[dict],
    results_directory=None,
    caption: str | None = None,
    label: str | None = None,
    decimals: int = 3,
) -> dict[str, Path]:
    """Write ``{stem}_metrics.csv``, ``{stem}_metrics.json``, and ``{stem}_table.tex``."""
    if kind not in {"binary", "novel", "category"}:
        raise ValueError(f"Unknown table kind {kind!r}")
    directory = Path(results_directory) if results_directory is not None else results_dir()
    directory.mkdir(parents=True, exist_ok=True)

    if kind == "binary":
        tex = binary_detection_table(
            records,
            caption=caption or BINARY_CAPTION,
            label=label or "tab:binary-detection",
            decimals=decimals,
        )
        summary = summarize_binary_runs(records)
        csv_rows = [_binary_csv_row(row, decimals) for row in summary]
        payload = {"kind": kind, "stem": stem, "runs": records, "summary": summary}
    elif kind == "novel":
        tex = novel_jailbreak_table(
            records,
            caption=caption or NOVEL_CAPTION,
            label=label or "tab:novel-jailbreak",
            decimals=decimals,
        )
        csv_rows = [_novel_csv_row(row, decimals) for row in records]
        payload = {"kind": kind, "stem": stem, "rows": records}
    else:
        tex = jailbreak_type_table(
            records,
            caption=caption or CATEGORY_CAPTION,
            label=label or "tab:jailbreak-type",
            decimals=decimals,
        )
        csv_rows = [_category_csv_row(row, decimals) for row in records]
        payload = {"kind": kind, "stem": stem, "rows": records}

    csv_path = directory / f"{stem}_metrics.csv"
    json_path = directory / f"{stem}_metrics.json"
    tex_path = directory / f"{stem}_table.tex"
    _write_csv(csv_path, csv_rows)
    json_path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "tex": tex_path}


def _binary_csv_row(row: dict, decimals: int) -> dict:
    exported = {"Features": row["features"], "Model": row["model"]}
    for metric, title in zip(BINARY_METRICS, BINARY_HEADERS):
        exported[f"{title} Mean"] = _format_number(row[f"{metric}_mean"], decimals)
        exported[f"{title} Std"] = _format_number(row[f"{metric}_std"], decimals)
    return exported


def _novel_csv_row(row: dict, decimals: int) -> dict:
    return {
        "Held-out Jailbreak Type": row["held_out_type"],
        "AUC": _format_number(row["auc"], decimals),
        "Accuracy": _format_number(row["accuracy"], decimals),
        "FNR": _format_number(row["fnr"], decimals),
        "TPR": _format_number(row["tpr"], decimals),
    }


def _category_csv_row(row: dict, decimals: int) -> dict:
    return {
        "Jailbreak Type": row["jailbreak_type"],
        "Accuracy": _format_number(row["accuracy"], decimals),
        "F1": _format_number(row["f1"], decimals),
        "AUC": _format_number(row["auc"], decimals),
        "TPR": _format_number(row["tpr"], decimals),
        "FNR": _format_number(row["fnr"], decimals),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty metrics CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _table(column_spec, caption, label, header_rows, body_rows, cmidrules: str | None = None) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + _escape(caption) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + column_spec + "}",
        r"\toprule",
        _latex_row(header_rows[0]),
    ]
    if cmidrules:
        lines.append(cmidrules)
    for header in header_rows[1:]:
        lines.append(_latex_row(header))
    lines.append(r"\midrule")
    for row in body_rows:
        lines.append(_latex_row(row))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _latex_row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\"


def _format_number(value, decimals: int) -> str:
    if is_missing_number(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def _as_float(value, metric: str) -> float:
    if is_missing_number(value):
        raise ValueError(f"Binary run is missing a numeric {metric}")
    return float(value)


def _require(row: dict, keys) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise KeyError(f"Record is missing {missing}")


def _escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if is_missing_number(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_ready(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return value
