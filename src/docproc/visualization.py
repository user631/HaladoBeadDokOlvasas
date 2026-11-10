from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from docproc.statistics import LIX_LEVELS

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FILE_TYPE_COLORS = {"pdf": SERIES[0], "txt": SERIES[1], "csv": SERIES[2]}
FALLBACK_COLOR = "#8a8984"


def _new_figure(width: float, height: float) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    axes = figure.add_subplot()
    _style_axes(axes)
    return figure, axes


def _style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    axes.xaxis.label.set_color(TEXT_SECONDARY)
    axes.yaxis.label.set_color(TEXT_SECONDARY)
    axes.set_axisbelow(True)


def _set_title(axes: Axes, title: str, subtitle: str | None = None) -> None:
    axes.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, pad=18 if subtitle else 8)
    if subtitle:
        axes.text(0, 1.01, subtitle, transform=axes.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom")


def plot_top_words(
    frequencies: Sequence[tuple[str, int]],
    title: str = "Leggyakoribb szavak",
    subtitle: str | None = None,
) -> Figure:
    words = [word for word, _ in frequencies][::-1]
    counts = [count for _, count in frequencies][::-1]
    figure, axes = _new_figure(7, max(2.5, 0.32 * len(words) + 1.2))
    axes.barh(words, counts, height=0.62, color=SERIES[0])
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_xlabel("előfordulás")
    axes.margins(y=0.02)
    _set_title(axes, title, subtitle)
    return figure


def plot_word_length_distribution(
    words: Sequence[str],
    title: str = "Szóhosszok eloszlása",
    subtitle: str | None = None,
) -> Figure:
    lengths = pd.Series([len(word) for word in words], dtype="int64")
    counts = lengths.value_counts().sort_index()
    figure, axes = _new_figure(7, 3.6)
    axes.bar(counts.index, counts.values, width=0.8, color=SERIES[0])
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_xlabel("szóhossz (karakter)")
    axes.set_ylabel("szavak száma")
    if not counts.empty:
        mean = lengths.mean()
        axes.set_ylim(top=counts.max() * 1.18)
        axes.axvline(mean, color=TEXT_SECONDARY, linewidth=1.2, linestyle="--")
        axes.text(mean, counts.max() * 1.16, f" átlag: {mean:.1f}", color=TEXT_SECONDARY, fontsize=9, va="top")
    _set_title(axes, title, subtitle)
    return figure


def plot_sentence_length_distribution(
    lengths: Sequence[int],
    title: str = "Mondathosszok eloszlása",
    subtitle: str | None = None,
    bins: int = 15,
) -> Figure:
    figure, axes = _new_figure(7, 3.6)
    if lengths:
        axes.hist(lengths, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_xlabel("mondathossz (szó)")
    axes.set_ylabel("mondatok száma")
    _set_title(axes, title, subtitle)
    return figure


def _type_legend(axes: Axes, file_types: Sequence[str]) -> None:
    present = [file_type for file_type in FILE_TYPE_COLORS if file_type in set(file_types)]
    handles = [Patch(color=FILE_TYPE_COLORS[file_type], label=file_type.upper()) for file_type in present]
    if len(handles) >= 2:
        legend = axes.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)


def plot_corpus_overview(
    stats: pd.DataFrame,
    column: str = "word_count",
    title: str = "Dokumentumok terjedelme",
    xlabel: str = "szavak száma",
    subtitle: str | None = None,
) -> Figure:
    ordered = stats.sort_values(column)
    colors = [FILE_TYPE_COLORS.get(file_type, FALLBACK_COLOR) for file_type in ordered["file_type"]]
    figure, axes = _new_figure(7, max(2.5, 0.4 * len(ordered) + 1.2))
    axes.barh(ordered.index, ordered[column], height=0.62, color=colors)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_xlabel(xlabel)
    _type_legend(axes, list(ordered["file_type"]))
    _set_title(axes, title, subtitle)
    return figure


def plot_readability(stats: pd.DataFrame, title: str = "Olvashatóság (LIX index)") -> Figure:
    figure = plot_corpus_overview(
        stats,
        column="lix",
        title=title,
        xlabel="LIX index (magasabb = nehezebb szöveg)",
        subtitle="sávhatárok: 30 / 40 / 50 / 60 (nagyon könnyű → nagyon nehéz)",
    )
    axes = figure.axes[0]
    for boundary, _ in LIX_LEVELS[:-1]:
        axes.axvline(boundary, color=GRID, linewidth=1.2, linestyle="--", zorder=0)
    return figure


def save_figure(figure: Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=SURFACE, metadata={"Software": None})
    return path
