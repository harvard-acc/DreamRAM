#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import webbrowser
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTickFormatter,
    BasicTicker,
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    CustomJSTickFormatter,
    DataTable,
    Div,
    FixedTicker,
    LinearColorMapper,
    Range1d,
    NumericInput,
    Select,
    TableColumn,
    TextAreaInput,
)
from bokeh.palettes import Cividis256, Greys256, Inferno256, Magma256, Plasma256, Viridis256
from bokeh.plotting import figure
from bokeh.server.server import Server

DEFAULT_MIN_ALPHA = 255
DEFAULT_MARGIN_FRAC = 0.07
DEFAULT_PALETTE = Viridis256
PALETTE_OPTIONS = {
    "viridis": Viridis256,
    "cividis": Cividis256,
    "plasma": Plasma256,
    "inferno": Inferno256,
    "magma": Magma256,
}
DEFAULT_POINT_RADIUS = 1
DEFAULT_UNGROUPED_NAME = "Ungrouped"
DEFAULT_ALL_COLUMNS_GROUP_NAME = "All columns"
DEFAULT_ALWAYS_VISIBLE_GROUP_NAME = "Always visible"
DEFAULT_NONE_GROUP_NAME = "Selection"
NONE_SORT_COLUMN_LABEL = "(none)"
LOG_CHECK_X = 0
LOG_CHECK_Y = 1
LOG_CHECK_COLOR = 2
BEST_MATCH_POINT_SIZE = 30
PARETO_POINT_SIZE = 8
BASELINE_POINT_SIZE = 16
AXIS_LABEL_FONT_SIZE = "24pt"
TICK_LABEL_FONT_SIZE = "14pt"
COLOR_BAR_TITLE_FONT_SIZE = "24pt"
COLOR_BAR_TICK_FONT_SIZE = "14pt"
JSON_PANEL_WIDTH = 360
MAX_BACKGROUND_CACHE_ENTRIES = 1
RENDER_CHUNK_SIZE = 1_000_000
BACKGROUND_GRAY_RGB = (188, 188, 188)
BACKGROUND_ALPHA_MIN = 255
BACKGROUND_ALPHA_MAX = 255
USER_PARETO_COLUMN = "user"
USER_PARETO_LINE_COLORS = {
    2: (68, 1, 84),
    4: (59, 82, 139),
    8: (33, 145, 140),
    16: (94, 201, 98),
    32: (253, 231, 37),
}

USER_PARETO_LINE_WIDTH = 3.6
USER_PARETO_POINT_LINE_WIDTH = 2.2
USER_PARETO_GREYS_LOW_FRACTION = 0.15

LOG2_TICK_JS = r"""
const rounded_tick = Math.round(tick)
if (Math.abs(tick - rounded_tick) > 1e-6) {
  return ""
}
const value = Math.pow(2, rounded_tick)
if (!isFinite(value)) {
  return ""
}
const absval = Math.abs(value)
let text = ""
if (absval === 0) {
  text = "0"
} else if (absval >= 1e6 || absval < 1e-4) {
  text = value.toExponential(3)
} else if (absval >= 1000) {
  text = value.toFixed(0)
} else if (absval >= 1) {
  text = value.toFixed(3)
} else {
  text = value.toPrecision(4)
}
return text.replace(/(\.\d*?[1-9])0+$/, "$1").replace(/\.0+$/, "").replace(/e\+/, "e")
"""


def _hex_palette_to_rgb(palette: list[str]) -> np.ndarray:
    rgb = np.empty((len(palette), 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        color = color.lstrip("#")
        rgb[i, 0] = int(color[0:2], 16)
        rgb[i, 1] = int(color[2:4], 16)
        rgb[i, 2] = int(color[4:6], 16)
    return rgb


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def _resample_palette_segment(
    palette: list[str],
    low_fraction: float = 0.0,
    high_fraction: float = 1.0,
    n_samples: int = 256,
) -> list[str]:
    if not palette:
        return []
    low_fraction = float(np.clip(low_fraction, 0.0, 1.0))
    high_fraction = float(np.clip(high_fraction, low_fraction, 1.0))
    max_index = len(palette) - 1
    start = low_fraction * max_index
    stop = high_fraction * max_index
    sample_positions = np.linspace(start, stop, int(max(1, n_samples)))
    sample_indices = np.clip(np.round(sample_positions).astype(np.int64), 0, max_index)
    return [palette[int(idx)] for idx in sample_indices]


def user_pareto_line_color(user_value: float) -> str:
    rounded = int(round(float(user_value)))
    if np.isfinite(user_value) and abs(float(user_value) - rounded) <= 1e-9:
        rgb = USER_PARETO_LINE_COLORS.get(rounded)
        if rgb is not None:
            return _rgb_to_hex(rgb)
    return "black"


@dataclass(frozen=True)
class DataStore:
    columns: list[str]
    arrays: dict[str, np.ndarray]
    matrix: np.ndarray
    row_count: int
    n_columns: int
    dtype_name: str
    load_seconds: float
    engine_used: str
    bytes_in_memory: int


@dataclass(frozen=True)
class LabelManager:
    columns: list[str]
    raw_to_friendly: dict[str, str]
    raw_to_display: dict[str, str]
    display_to_raw: dict[str, str]
    unique_friendly_to_raw: dict[str, str]

    @classmethod
    def build(cls, columns: list[str], label_csv: Path | None) -> "LabelManager":
        raw_to_friendly = {column: column for column in columns}

        if label_csv is not None:
            if not label_csv.exists():
                raise FileNotFoundError(f"Label CSV not found: {label_csv}")
            with label_csv.open("r", newline="") as handle:
                reader = csv.reader(handle)
                for row_idx, row in enumerate(reader, start=1):
                    if not row or all(str(cell).strip() == "" for cell in row):
                        continue
                    if len(row) < 2:
                        raise ValueError(
                            f"Label CSV row {row_idx} must have at least two columns: raw_header,friendly_label"
                        )
                    raw = str(row[0]).strip()
                    friendly = str(row[1]).strip()
                    if raw not in raw_to_friendly:
                        raise ValueError(f"Label CSV row {row_idx} refers to unknown data column: {raw}")
                    if friendly:
                        raw_to_friendly[raw] = friendly

        friendly_counts = Counter(raw_to_friendly.values())
        raw_to_display: dict[str, str] = {}
        display_to_raw: dict[str, str] = {}
        unique_friendly_to_raw: dict[str, str] = {}

        for raw in columns:
            friendly = raw_to_friendly[raw]
            if friendly_counts[friendly] == 1:
                display = friendly
                unique_friendly_to_raw[friendly] = raw
            else:
                display = f"{friendly} [{raw}]"
            raw_to_display[raw] = display
            display_to_raw[display] = raw

        return cls(
            columns=list(columns),
            raw_to_friendly=raw_to_friendly,
            raw_to_display=raw_to_display,
            display_to_raw=display_to_raw,
            unique_friendly_to_raw=unique_friendly_to_raw,
        )

    def options(self, subset: list[str] | None = None, include_none: bool = False) -> list[tuple[str, str]]:
        raw_columns = subset if subset is not None else self.columns
        values: list[tuple[str, str]] = []
        if include_none:
            values.append(("", NONE_SORT_COLUMN_LABEL))
        values.extend((raw, self.raw_to_display[raw]) for raw in raw_columns)
        return values

    def label(self, raw_column: str) -> str:
        return self.raw_to_display[raw_column]

    def resolve(self, ref: str) -> str:
        key = str(ref).strip()
        if key in self.raw_to_display:
            return key
        if key in self.display_to_raw:
            return self.display_to_raw[key]
        if key in self.unique_friendly_to_raw:
            return self.unique_friendly_to_raw[key]
        raise KeyError(
            f"Unknown column reference: {ref!r}. Use a raw header, a unique friendly label, or an exact dropdown label."
        )


@dataclass(frozen=True)
class ColumnGroupManager:
    columns: list[str]
    subgroup_order: list[str]
    subgroup_to_columns: dict[str, list[str]]
    column_to_subgroup: dict[str, str]
    always_visible_columns: list[str]
    always_visible_group_name: str
    ungrouped_name: str

    @classmethod
    def build(cls, columns: list[str], subgroup_json: Path | None) -> "ColumnGroupManager":
        payload: dict[str, object]
        if subgroup_json is None:
            payload = {}
            ungrouped_default = DEFAULT_ALL_COLUMNS_GROUP_NAME
        else:
            if not subgroup_json.exists():
                raise FileNotFoundError(f"Subgroups JSON not found: {subgroup_json}")
            with subgroup_json.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("Subgroups JSON must be a JSON object.")
            payload = loaded
            ungrouped_default = DEFAULT_UNGROUPED_NAME

        ungrouped_name = str(payload.get("ungrouped_name", ungrouped_default)).strip() or ungrouped_default

        has_explicit_always_visible = "always_visible" in payload
        raw_always_visible = payload.get("always_visible", [])
        if raw_always_visible is None:
            raw_always_visible = []
        if not isinstance(raw_always_visible, list):
            raise ValueError("Subgroups JSON field 'always_visible' must be a list of raw column names.")

        always_visible_group_name = (
            str(payload.get("always_visible_group_name", DEFAULT_ALWAYS_VISIBLE_GROUP_NAME)).strip()
            or DEFAULT_ALWAYS_VISIBLE_GROUP_NAME
        )

        always_visible_columns: list[str] = []
        always_visible_set: set[str] = set()
        for entry in raw_always_visible:
            raw = str(entry).strip()
            if raw == "":
                continue
            if raw not in columns:
                raise ValueError(f"Subgroups JSON 'always_visible' refers to unknown data column: {raw}")
            if raw in always_visible_set:
                raise ValueError(f"Subgroups JSON 'always_visible' contains the same column more than once: {raw}")
            always_visible_set.add(raw)
            always_visible_columns.append(raw)

        if not has_explicit_always_visible and "id" in columns and "id" not in always_visible_set:
            always_visible_columns.append("id")
            always_visible_set.add("id")

        raw_subgroups = payload.get("subgroups", [])
        if raw_subgroups is None:
            raw_subgroups = []
        if not isinstance(raw_subgroups, list):
            raise ValueError("Subgroups JSON field 'subgroups' must be a list.")

        subgroup_order: list[str] = []
        subgroup_to_columns: dict[str, list[str]] = {}
        column_to_subgroup: dict[str, str] = {}
        seen_group_names: set[str] = set()
        assigned_columns: set[str] = set()

        for group_idx, group_item in enumerate(raw_subgroups, start=1):
            if not isinstance(group_item, dict):
                raise ValueError(f"Subgroup entry {group_idx} must be an object.")

            name = str(group_item.get("name", "")).strip()
            if not name:
                raise ValueError(f"Subgroup entry {group_idx} is missing a non-empty 'name'.")
            if name in seen_group_names:
                raise ValueError(f"Subgroups JSON contains duplicate subgroup name: {name}")
            seen_group_names.add(name)

            raw_columns = group_item.get("columns")
            if not isinstance(raw_columns, list):
                raise ValueError(f"Subgroup '{name}' must have a 'columns' list.")

            parsed_columns: list[str] = []
            seen_within_group: set[str] = set()
            for entry in raw_columns:
                raw = str(entry).strip()
                if raw == "":
                    continue
                if raw not in columns:
                    raise ValueError(f"Subgroup '{name}' refers to unknown data column: {raw}")
                if raw in seen_within_group:
                    raise ValueError(f"Subgroup '{name}' contains the same column more than once: {raw}")
                if raw in assigned_columns:
                    raise ValueError(f"Data column '{raw}' appears in more than one subgroup.")
                seen_within_group.add(raw)
                assigned_columns.add(raw)
                parsed_columns.append(raw)
                column_to_subgroup[raw] = name

            subgroup_order.append(name)
            subgroup_to_columns[name] = parsed_columns

        ungrouped_columns = [column for column in columns if column not in assigned_columns]
        if ungrouped_columns:
            subgroup_order.append(ungrouped_name)
            subgroup_to_columns[ungrouped_name] = ungrouped_columns
            for column in ungrouped_columns:
                column_to_subgroup[column] = ungrouped_name

        return cls(
            columns=list(columns),
            subgroup_order=subgroup_order,
            subgroup_to_columns=subgroup_to_columns,
            column_to_subgroup=column_to_subgroup,
            always_visible_columns=always_visible_columns,
            always_visible_group_name=always_visible_group_name,
            ungrouped_name=ungrouped_name,
        )

    def grouped_options(
        self,
        labels: LabelManager,
        include_none: bool = False,
        none_group_name: str = DEFAULT_NONE_GROUP_NAME,
    ) -> dict[str, list[tuple[str, str]]]:
        grouped: dict[str, list[tuple[str, str]]] = {}
        if include_none:
            grouped[none_group_name] = [("", NONE_SORT_COLUMN_LABEL)]
        if self.always_visible_columns:
            grouped[self.always_visible_group_name] = [
                (column, labels.label(column)) for column in self.always_visible_columns
            ]
        always_visible = set(self.always_visible_columns)
        for group_name in self.subgroup_order:
            columns = [
                column for column in self.subgroup_to_columns.get(group_name, []) if column not in always_visible
            ]
            if not columns:
                continue
            grouped[group_name] = [(column, labels.label(column)) for column in columns]
        return grouped

    def toggleable_groups(self) -> list[str]:
        toggleable: list[str] = []
        always_visible = set(self.always_visible_columns)
        for group_name in self.subgroup_order:
            if any(column not in always_visible for column in self.subgroup_to_columns.get(group_name, [])):
                toggleable.append(group_name)
        return toggleable


@dataclass(frozen=True)
class BaselineInfo:
    index: int
    column_title: str
    summary_label: str


@dataclass(frozen=True)
class SortSpec:
    column: str
    direction: str


@dataclass
class FilterRow:
    column_select: Select
    min_input: NumericInput
    max_input: NumericInput
    remove_button: Button
    info: Div
    container: object


@dataclass
class SortControl:
    column_select: Select
    direction_select: Select


@dataclass(frozen=True)
class ParetoResult:
    enabled: bool
    relative_indices: np.ndarray
    display_relative_indices: np.ndarray


@dataclass(frozen=True)
class ParetoSubsetResult:
    group_value: float
    frontier_plot_positions: np.ndarray
    display_plot_positions: np.ndarray
    frontier_row_indices: np.ndarray


@dataclass(frozen=True)
class BestRowDescriptor:
    row_index: int
    color_value: float
    group_column: str | None = None
    group_value: float | None = None


class StatsCache:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self._cache: dict[str, tuple[float, float]] = {}

    def get(self, column: str) -> tuple[float, float]:
        if column not in self._cache:
            arr = self.arrays[column]
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                self._cache[column] = (0.0, 1.0)
            else:
                self._cache[column] = (float(finite.min()), float(finite.max()))
        return self._cache[column]


class FilterMaskCache:
    def __init__(self, arrays: dict[str, np.ndarray], row_count: int):
        self.arrays = arrays
        self.row_count = int(row_count)
        self._signature: tuple[tuple[str, float | None, float | None], ...] | None = None
        self._mask: np.ndarray | None = None
        self._all_true = np.ones(self.row_count, dtype=bool)

    def get(self, filters: list[tuple[str, float | None, float | None]]) -> np.ndarray:
        signature = tuple((column, lo, hi) for column, lo, hi in filters)
        if self._signature == signature and self._mask is not None:
            return self._mask

        if not filters:
            self._signature = signature
            self._mask = self._all_true
            return self._mask

        mask = np.ones(self.row_count, dtype=bool)
        for column, lo, hi in filters:
            arr = self.arrays[column]
            mask &= np.isfinite(arr)
            if lo is not None:
                mask &= arr >= lo
            if hi is not None:
                mask &= arr <= hi
        self._signature = signature
        self._mask = mask
        return mask


class EmptySelectionError(RuntimeError):
    pass



@dataclass(frozen=True)
class PlotBounds:
    x_img_min: float
    x_img_max: float
    y_img_min: float
    y_img_max: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]


@dataclass(frozen=True)
class BackgroundCacheEntry:
    image: np.ndarray
    bounds: PlotBounds
    n_rows: int
    n_nonzero_pixels: int
    max_count_per_pixel: int


class PlotComputer:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        width: int,
        height: int,
        palette_rgb: np.ndarray,
        min_alpha: int = DEFAULT_MIN_ALPHA,
        point_radius: int = DEFAULT_POINT_RADIUS,
        chunk_size: int = RENDER_CHUNK_SIZE,
    ):
        self.arrays = arrays
        self.width = int(width)
        self.height = int(height)
        self.min_alpha = int(min_alpha)
        self.point_radius = int(point_radius)
        self.palette_rgb = np.ascontiguousarray(palette_rgb, dtype=np.uint8)
        self.chunk_size = max(1, int(chunk_size))
        self.n_pixels = self.width * self.height
        self.empty_image = np.zeros((self.height, self.width), dtype=np.uint32)

    def build_xy_mask(
        self,
        filter_mask: np.ndarray,
        x_col: str,
        y_col: str,
        x_log: bool,
        y_log: bool,
    ) -> np.ndarray:
        x = self.arrays[x_col]
        y = self.arrays[y_col]
        mask = filter_mask.copy()
        mask &= np.isfinite(x)
        mask &= np.isfinite(y)
        if x_log:
            mask &= x > 0
        if y_log:
            mask &= y > 0
        return mask

    def build_plot_mask(
        self,
        filter_mask: np.ndarray,
        x_col: str,
        y_col: str,
        color_col: str,
        x_log: bool,
        y_log: bool,
        color_log: bool,
    ) -> np.ndarray:
        mask = self.build_xy_mask(filter_mask, x_col, y_col, x_log, y_log)
        c = self.arrays[color_col]
        mask &= np.isfinite(c)
        if color_log:
            mask &= c > 0
        return mask

    def _iter_mask_slices(self, mask: np.ndarray):
        idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        if idx.size == 0:
            return
        step = self.chunk_size
        for start in range(0, idx.size, step):
            yield idx[start : start + step]

    def compute_bounds_from_mask(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        mask: np.ndarray,
        x_log: bool,
        y_log: bool,
    ) -> PlotBounds:
        found = False
        x_min = x_max = y_min = y_max = 0.0
        for idx in self._iter_mask_slices(mask):
            x_chunk = transform_values_inplace(x_values[idx], x_log)
            y_chunk = transform_values_inplace(y_values[idx], y_log)
            chunk_x_min = float(np.min(x_chunk))
            chunk_x_max = float(np.max(x_chunk))
            chunk_y_min = float(np.min(y_chunk))
            chunk_y_max = float(np.max(y_chunk))
            if not found:
                x_min, x_max = chunk_x_min, chunk_x_max
                y_min, y_max = chunk_y_min, chunk_y_max
                found = True
            else:
                x_min = min(x_min, chunk_x_min)
                x_max = max(x_max, chunk_x_max)
                y_min = min(y_min, chunk_y_min)
                y_max = max(y_max, chunk_y_max)
        if not found:
            raise EmptySelectionError("No plottable rows remain after filters and log-scale requirements.")
        x_img_min, x_img_max, y_img_min, y_img_max = _match_extent_aspect(
            x_min,
            x_max,
            y_min,
            y_max,
            self.width,
            self.height,
        )
        return PlotBounds(
            x_img_min=x_img_min,
            x_img_max=x_img_max,
            y_img_min=y_img_min,
            y_img_max=y_img_max,
            x_range=_padded_range(x_min, x_max, frac=DEFAULT_MARGIN_FRAC),
            y_range=_padded_range(y_min, y_max, frac=DEFAULT_MARGIN_FRAC),
        )

    def render_background_from_mask(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        mask: np.ndarray,
        x_log: bool,
        y_log: bool,
    ) -> BackgroundCacheEntry:
        bounds = self.compute_bounds_from_mask(x_values, y_values, mask, x_log, y_log)
        counts = np.zeros(self.n_pixels, dtype=np.uint32)
        n_rows = 0
        for idx in self._iter_mask_slices(mask):
            x_chunk = transform_values_inplace(x_values[idx], x_log)
            y_chunk = transform_values_inplace(y_values[idx], y_log)
            flat = self._coords_to_flat_pixels(
                x_chunk,
                y_chunk,
                bounds.x_img_min,
                bounds.x_img_max,
                bounds.y_img_min,
                bounds.y_img_max,
            )
            counts += np.bincount(flat, minlength=self.n_pixels).astype(np.uint32, copy=False)
            n_rows += int(idx.size)
        counts_vis = self._expand_counts(counts)
        image = self._counts_to_gray_rgba(counts_vis.ravel())
        return BackgroundCacheEntry(
            image=image,
            bounds=bounds,
            n_rows=n_rows,
            n_nonzero_pixels=int(np.count_nonzero(counts)),
            max_count_per_pixel=int(np.max(counts)) if counts.size else 0,
        )

    def render_color_overlay_from_indices(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        c_values: np.ndarray,
        indices: np.ndarray,
        x_log: bool,
        y_log: bool,
        color_log: bool,
        z_order: str,
        bounds: PlotBounds,
        color_low: float,
        color_high: float,
    ) -> dict[str, object]:
        if indices.size == 0:
            return {
                "image": self.empty_image,
                "n_rows": 0,
                "n_nonzero_pixels": 0,
                "max_count_per_pixel": 0,
            }
        counts = np.zeros(self.n_pixels, dtype=np.uint32)
        fill_value = -np.inf if z_order == "highest" else np.inf
        color_top = np.full(self.n_pixels, fill_value, dtype=np.float32)
        n_rows = 0
        step = self.chunk_size
        for start in range(0, indices.size, step):
            idx = indices[start : start + step]
            x_chunk = transform_values_inplace(x_values[idx], x_log)
            y_chunk = transform_values_inplace(y_values[idx], y_log)
            c_chunk = transform_values_inplace(c_values[idx], color_log)
            flat = self._coords_to_flat_pixels(
                x_chunk,
                y_chunk,
                bounds.x_img_min,
                bounds.x_img_max,
                bounds.y_img_min,
                bounds.y_img_max,
            )
            counts += np.bincount(flat, minlength=self.n_pixels).astype(np.uint32, copy=False)
            if z_order == "highest":
                np.maximum.at(color_top, flat, c_chunk)
            else:
                np.minimum.at(color_top, flat, c_chunk)
            n_rows += int(idx.size)
        counts_vis, color_vis = self._expand_points(counts, color_top, z_order)
        image = self._colors_to_rgba(counts_vis.ravel(), color_vis.ravel(), color_low, color_high)
        return {
            "image": image,
            "n_rows": n_rows,
            "n_nonzero_pixels": int(np.count_nonzero(counts)),
            "max_count_per_pixel": int(np.max(counts)) if counts.size else 0,
        }

    def compute_bounds(self, x_plot: np.ndarray, y_plot: np.ndarray) -> PlotBounds:
        if x_plot.size == 0:
            raise EmptySelectionError("No plottable rows remain after filters and log-scale requirements.")
        x_min = float(x_plot.min())
        x_max = float(x_plot.max())
        y_min = float(y_plot.min())
        y_max = float(y_plot.max())
        x_img_min, x_img_max, y_img_min, y_img_max = _match_extent_aspect(
            x_min,
            x_max,
            y_min,
            y_max,
            self.width,
            self.height,
        )
        return PlotBounds(
            x_img_min=x_img_min,
            x_img_max=x_img_max,
            y_img_min=y_img_min,
            y_img_max=y_img_max,
            x_range=_padded_range(x_min, x_max, frac=DEFAULT_MARGIN_FRAC),
            y_range=_padded_range(y_min, y_max, frac=DEFAULT_MARGIN_FRAC),
        )

    def render_background(self, x_plot: np.ndarray, y_plot: np.ndarray) -> BackgroundCacheEntry:
        bounds = self.compute_bounds(x_plot, y_plot)
        counts = self._accumulate_counts(x_plot, y_plot, bounds)
        counts_vis = self._expand_counts(counts)
        image = self._counts_to_gray_rgba(counts_vis.ravel())
        return BackgroundCacheEntry(
            image=image,
            bounds=bounds,
            n_rows=int(x_plot.size),
            n_nonzero_pixels=int(np.count_nonzero(counts)),
            max_count_per_pixel=int(np.max(counts)) if counts.size else 0,
        )

    def render_color_overlay(
        self,
        x_plot: np.ndarray,
        y_plot: np.ndarray,
        c_plot: np.ndarray,
        z_order: str,
        bounds: PlotBounds,
        color_low: float,
        color_high: float,
    ) -> dict[str, object]:
        if x_plot.size == 0:
            return {
                "image": self.empty_image,
                "n_rows": 0,
                "n_nonzero_pixels": 0,
                "max_count_per_pixel": 0,
            }

        flat = self._coords_to_flat_pixels(
            x_plot,
            y_plot,
            bounds.x_img_min,
            bounds.x_img_max,
            bounds.y_img_min,
            bounds.y_img_max,
        )
        counts = np.bincount(flat, minlength=self.n_pixels).astype(np.uint32, copy=False)

        fill_value = -np.inf if z_order == "highest" else np.inf
        color_top = np.full(self.n_pixels, fill_value, dtype=np.float32)
        colors = c_plot.astype(np.float32, copy=False)
        if z_order == "highest":
            np.maximum.at(color_top, flat, colors)
        else:
            np.minimum.at(color_top, flat, colors)

        counts_vis, color_vis = self._expand_points(counts, color_top, z_order)
        image = self._colors_to_rgba(counts_vis.ravel(), color_vis.ravel(), color_low, color_high)
        return {
            "image": image,
            "n_rows": int(x_plot.size),
            "n_nonzero_pixels": int(np.count_nonzero(counts)),
            "max_count_per_pixel": int(np.max(counts)) if counts.size else 0,
        }

    def _coords_to_flat_pixels(
        self,
        x_plot: np.ndarray,
        y_plot: np.ndarray,
        x_img_min: float,
        x_img_max: float,
        y_img_min: float,
        y_img_max: float,
    ) -> np.ndarray:
        x_scale = (self.width - 1) / (x_img_max - x_img_min)
        y_scale = (self.height - 1) / (y_img_max - y_img_min)
        px = ((x_plot - x_img_min) * x_scale).astype(np.int32)
        py = ((y_plot - y_img_min) * y_scale).astype(np.int32)
        np.clip(px, 0, self.width - 1, out=px)
        np.clip(py, 0, self.height - 1, out=py)
        return py.astype(np.int32, copy=False) * np.int32(self.width) + px.astype(np.int32, copy=False)

    def coords_to_pixels(
        self,
        x_plot: np.ndarray,
        y_plot: np.ndarray,
        x_img_min: float,
        x_img_max: float,
        y_img_min: float,
        y_img_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        flat = self._coords_to_flat_pixels(x_plot, y_plot, x_img_min, x_img_max, y_img_min, y_img_max)
        py = (flat // self.width).astype(np.int32, copy=False)
        px = (flat - py * self.width).astype(np.int32, copy=False)
        return px, py

    def _accumulate_counts(self, x_plot: np.ndarray, y_plot: np.ndarray, bounds: PlotBounds) -> np.ndarray:
        flat = self._coords_to_flat_pixels(
            x_plot,
            y_plot,
            bounds.x_img_min,
            bounds.x_img_max,
            bounds.y_img_min,
            bounds.y_img_max,
        )
        counts = np.bincount(flat, minlength=self.n_pixels).astype(np.uint32, copy=False)
        return counts

    def _expand_counts(self, counts_flat: np.ndarray) -> np.ndarray:
        counts_2d = counts_flat.reshape(self.height, self.width)
        if self.point_radius <= 0:
            return counts_2d
        return self._square_filter(counts_2d, mode="max", fill_value=0)

    def _expand_points(
        self,
        counts_flat: np.ndarray,
        color_flat: np.ndarray,
        z_order: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        counts_2d = counts_flat.reshape(self.height, self.width)
        color_2d = color_flat.reshape(self.height, self.width)
        if self.point_radius <= 0:
            return counts_2d, color_2d
        counts_vis = self._square_filter(counts_2d, mode="max", fill_value=0)
        color_fill = -np.inf if z_order == "highest" else np.inf
        color_mode = "max" if z_order == "highest" else "min"
        color_vis = self._square_filter(color_2d, mode=color_mode, fill_value=color_fill)
        return counts_vis, color_vis

    def _square_filter(
        self,
        arr: np.ndarray,
        mode: str,
        fill_value: int | float,
        radius: int | None = None,
    ) -> np.ndarray:
        radius = self.point_radius if radius is None else int(radius)
        if radius <= 0:
            return arr
        height, width = arr.shape
        out = np.full(arr.shape, fill_value, dtype=arr.dtype)
        for dy in range(-radius, radius + 1):
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            for dx in range(-radius, radius + 1):
                src_x0 = max(0, -dx)
                src_x1 = min(width, width - dx)
                dst_x0 = max(0, dx)
                dst_x1 = min(width, width + dx)
                src = arr[src_y0:src_y1, src_x0:src_x1]
                dst = out[dst_y0:dst_y1, dst_x0:dst_x1]
                if mode == "max":
                    np.maximum(dst, src, out=dst)
                elif mode == "min":
                    np.minimum(dst, src, out=dst)
                else:
                    raise ValueError(f"Unsupported filter mode: {mode}")
        return out

    def _counts_to_gray_rgba(self, counts: np.ndarray) -> np.ndarray:
        img = np.zeros(self.n_pixels, dtype=np.uint32)
        nonzero = counts > 0
        if not np.any(nonzero):
            return img.reshape(self.height, self.width)
        alpha = np.full(int(np.count_nonzero(nonzero)), BACKGROUND_ALPHA_MAX, dtype=np.uint8)
        view = img.view(np.uint8).reshape(-1, 4)
        view[nonzero, 0] = BACKGROUND_GRAY_RGB[0]
        view[nonzero, 1] = BACKGROUND_GRAY_RGB[1]
        view[nonzero, 2] = BACKGROUND_GRAY_RGB[2]
        view[nonzero, 3] = alpha
        return img.reshape(self.height, self.width)

    def _colors_to_rgba(
        self,
        counts: np.ndarray,
        color_top: np.ndarray,
        color_low: float,
        color_high: float,
    ) -> np.ndarray:
        img = np.zeros(self.n_pixels, dtype=np.uint32)
        nonzero = counts > 0
        if not np.any(nonzero):
            return img.reshape(self.height, self.width)
        low = float(color_low)
        high = float(color_high)
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 1.0
        if low == high:
            low -= 0.5
            high += 0.5
        selected_colors = color_top[nonzero]
        norm = (selected_colors - low) / (high - low)
        palette_idx = np.clip((norm * (len(self.palette_rgb) - 1)).astype(np.int32), 0, len(self.palette_rgb) - 1)
        alpha = np.full(int(np.count_nonzero(nonzero)), 255, dtype=np.uint8)
        view = img.view(np.uint8).reshape(-1, 4)
        view[nonzero, 0] = self.palette_rgb[palette_idx, 0]
        view[nonzero, 1] = self.palette_rgb[palette_idx, 1]
        view[nonzero, 2] = self.palette_rgb[palette_idx, 2]
        view[nonzero, 3] = alpha
        return img.reshape(self.height, self.width)


def _ensure_nonzero_extent(vmin: float, vmax: float) -> tuple[float, float]:
    if vmin == vmax:
        pad = 0.5 if vmin == 0 else max(abs(vmin) * 0.01, 1e-12)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _padded_range(vmin: float, vmax: float, frac: float) -> tuple[float, float]:
    span = vmax - vmin
    if span <= 0:
        span = 1.0 if vmin == 0 else max(abs(vmin) * 0.1, 1e-12)
    pad = span * frac
    return vmin - pad, vmax + pad


def _match_extent_aspect(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x_min, x_max = _ensure_nonzero_extent(x_min, x_max)
    y_min, y_max = _ensure_nonzero_extent(y_min, y_max)

    x_span = x_max - x_min
    y_span = y_max - y_min
    target_ratio = float(width) / float(height)
    current_ratio = x_span / y_span

    if np.isclose(current_ratio, target_ratio, rtol=1e-9, atol=1e-12):
        return x_min, x_max, y_min, y_max

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    if current_ratio < target_ratio:
        x_span = y_span * target_ratio
    else:
        y_span = x_span / target_ratio

    return (
        x_center - 0.5 * x_span,
        x_center + 0.5 * x_span,
        y_center - 0.5 * y_span,
        y_center + 0.5 * y_span,
    )


def load_float_csv(csv_path: Path, dtype: str = "float32") -> DataStore:
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    np_dtype = np.float32 if dtype == "float32" else np.float64

    t0 = time.perf_counter()
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty.") from exc
    columns = [str(col).strip() for col in header]
    if not columns or any(col == "" for col in columns):
        raise ValueError("CSV header contains an empty column name.")
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header contains duplicate column names.")

    matrix = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np_dtype, ndmin=2)
    load_seconds = time.perf_counter() - t0

    if matrix.size == 0 or matrix.shape[0] == 0:
        raise ValueError("CSV loaded successfully, but it contains no rows.")
    if matrix.shape[1] != len(columns):
        raise ValueError(
            f"CSV header has {len(columns)} columns, but parsed data has {matrix.shape[1]} columns."
        )

    print("Building Column Dictionary...")

    if not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix)

    arrays: dict[str, np.ndarray] = {str(columns[i]): matrix[:, i] for i in range(len(columns))}

    store = DataStore(
        columns=columns,
        arrays=arrays,
        matrix=matrix,
        row_count=int(matrix.shape[0]),
        n_columns=int(matrix.shape[1]),
        dtype_name=np.dtype(np_dtype).name,
        load_seconds=load_seconds,
        engine_used="numpy.loadtxt",
        bytes_in_memory=int(matrix.nbytes),
    )

    gc.collect()
    return store


def load_simple_axes_csv(simple_axes_csv: Path | None, columns: list[str]) -> list[str]:
    if simple_axes_csv is None:
        return list(columns)
    if not simple_axes_csv.exists():
        raise FileNotFoundError(f"Simple-axes CSV not found: {simple_axes_csv}")

    with simple_axes_csv.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            first_row = next(reader)
        except StopIteration as exc:
            raise ValueError("Simple-axes CSV is empty.") from exc

    selected: list[str] = []
    seen: set[str] = set()
    for entry in first_row:
        raw = str(entry).strip()
        if not raw:
            continue
        if raw not in columns:
            raise ValueError(f"Simple-axes CSV refers to unknown data column: {raw}")
        if raw not in seen:
            selected.append(raw)
            seen.add(raw)

    if not selected:
        raise ValueError("Simple-axes CSV must contain at least one valid data column.")
    return selected


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


def format_float_value(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{float(value):.12g}"


def is_integer_like(value: float) -> bool:
    if not np.isfinite(value):
        return False
    nearest = round(float(value))
    tolerance = 1e-9 * max(1.0, abs(float(value)))
    return abs(float(value) - nearest) <= tolerance



def format_plain_decimal(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    if is_integer_like(value):
        return f"{round(float(value)):.0f}"
    return np.format_float_positional(float(value), trim="-")


def format_decimal_places(value: float, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return str(value)
    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def format_best_row_value(column_name: str, value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    if column_name == "id" or is_integer_like(value):
        return format_plain_decimal(value)
    return format_decimal_places(value, decimals=3)


def parse_optional_float(value: object) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    return float(value)


def parse_filters_payload(payload: object, labels: LabelManager) -> list[tuple[str, float | None, float | None]]:
    if payload in (None, ""):
        return []
    if isinstance(payload, dict):
        if "filters" in payload and isinstance(payload["filters"], list):
            payload = payload["filters"]
        else:
            normalized: list[dict[str, object]] = []
            for column_ref, spec in payload.items():
                if isinstance(spec, dict):
                    normalized.append(
                        {
                            "column": column_ref,
                            "min": spec.get("min"),
                            "max": spec.get("max"),
                        }
                    )
                elif isinstance(spec, (list, tuple)):
                    normalized.append(
                        {
                            "column": column_ref,
                            "min": spec[0] if len(spec) >= 1 else None,
                            "max": spec[1] if len(spec) >= 2 else None,
                        }
                    )
                else:
                    raise ValueError(
                        "Dictionary-style filter JSON must map each column to either {min,max} or [min,max]."
                    )
            payload = normalized

    if not isinstance(payload, list):
        raise ValueError("Filter JSON must be a list of filter objects or a dictionary of column specs.")

    result: list[tuple[str, float | None, float | None]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Filter entry {idx} must be an object.")
        column_ref = item.get("column", item.get("label", item.get("name")))
        if column_ref is None:
            raise ValueError(f"Filter entry {idx} is missing a 'column' field.")
        raw = labels.resolve(str(column_ref))
        lo = parse_optional_float(item.get("min"))
        hi = parse_optional_float(item.get("max"))
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Filter entry {idx} has min > max.")
        result.append((raw, lo, hi))
    return result


def parse_filters_json(text: str, labels: LabelManager) -> list[tuple[str, float | None, float | None]]:
    if text.strip() == "":
        return []
    return parse_filters_payload(json.loads(text), labels)


def filters_to_json_payload(filters: list[tuple[str, float | None, float | None]]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for column, lo, hi in filters:
        item: dict[str, object] = {"column": column}
        if lo is not None:
            item["min"] = lo
        if hi is not None:
            item["max"] = hi
        payload.append(item)
    return payload


def filters_to_json_text(filters: list[tuple[str, float | None, float | None]]) -> str:
    return json.dumps(filters_to_json_payload(filters), indent=2)


def parse_bool_flag(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Field '{field_name}' must be true or false.")


def parse_min_max_direction(value: object, field_name: str) -> str:
    direction = str(value).strip().lower()
    if direction not in {"max", "min"}:
        raise ValueError(f"Field '{field_name}' must be 'max' or 'min'.")
    return direction


def format_log_tick_value(value: float) -> str:
    absval = abs(float(value))
    if absval == 0:
        return "0"
    if absval >= 1e6 or absval < 1e-4:
        text = f"{float(value):.3e}"
    elif absval >= 1000:
        text = f"{float(value):.0f}"
    elif absval >= 1:
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{float(value):.4g}"
    return text.replace("e+", "e")


def _nearest_or_floor(value: float, tol: float = 1e-12) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) <= tol:
        return rounded
    return int(np.floor(float(value)))



def _nearest_or_ceil(value: float, tol: float = 1e-12) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) <= tol:
        return rounded
    return int(np.ceil(float(value)))



def build_log_tick_values(low: float, high: float) -> list[float]:
    low_i = _nearest_or_floor(low)
    high_i = _nearest_or_ceil(high)
    if high_i < low_i:
        return []
    return [float(v) for v in range(low_i, high_i + 1)]



def build_log_axis_major_ticks(low: float, high: float) -> list[float]:
    low_i = int(np.ceil(low - 1e-12))
    high_i = int(np.floor(high + 1e-12))
    if high_i < low_i:
        return []
    return [float(v) for v in range(low_i, high_i + 1)]



def build_log_axis_minor_ticks(low: float, high: float) -> list[float]:
    minor_offsets = np.log2(np.array([1.2, 1.4, 1.6, 1.8], dtype=np.float64))
    start = int(np.floor(low - 1e-12))
    stop = int(np.ceil(high + 1e-12))
    ticks: list[float] = []
    for exponent in range(start, stop):
        for offset in minor_offsets:
            tick = float(exponent + offset)
            if low < tick < high:
                ticks.append(tick)
    return ticks



def make_log_axis_ticker(low: float, high: float) -> FixedTicker:
    return FixedTicker(
        ticks=build_log_axis_major_ticks(low, high),
        minor_ticks=build_log_axis_minor_ticks(low, high),
        num_minor_ticks=0,
    )



def expand_log_display_range_for_power_ticks(low: float, high: float) -> tuple[float, float]:
    low_i = _nearest_or_floor(low)
    high_i = _nearest_or_ceil(high)
    if low_i == high_i:
        center = float(low_i)
        return center - 0.5, center + 0.5
    return float(low_i), float(high_i)



def build_linear_power_tick_values(low: float, high: float) -> list[float]:
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= 0:
        return []
    low_i = _nearest_or_floor(np.log2(low))
    high_i = _nearest_or_ceil(np.log2(high))
    if high_i < low_i:
        return []
    return [float(2.0**v) for v in range(low_i, high_i + 1)]



def expand_linear_display_range_for_power_ticks(low: float, high: float) -> tuple[float, float]:
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= 0:
        if low == high:
            return low - 0.5, high + 0.5
        return low, high
    low_i = _nearest_or_floor(np.log2(low))
    high_i = _nearest_or_ceil(np.log2(high))
    if low_i == high_i:
        center = float(2.0**low_i)
        spread = float(np.sqrt(2.0))
        return center / spread, center * spread
    return float(2.0**low_i), float(2.0**high_i)


def normalize_sort_column_ref(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", NONE_SORT_COLUMN_LABEL}:
        return ""
    return text


def parse_sort_controls_payload(
    payload: object,
    labels: LabelManager,
    n_controls: int,
    default_directions: list[str] | None = None,
) -> list[SortSpec]:
    if payload is None:
        payload = []
    if not isinstance(payload, list):
        raise ValueError("'best_match_sort' must be a list of sort objects.")

    normalized_defaults = list(default_directions or [])
    while len(normalized_defaults) < n_controls:
        normalized_defaults.append("max")

    result: list[SortSpec] = []
    for idx in range(n_controls):
        direction_default = normalized_defaults[idx]
        if idx >= len(payload):
            result.append(SortSpec("", direction_default))
            continue

        item = payload[idx]
        if item is None:
            result.append(SortSpec("", direction_default))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Sort entry {idx + 1} must be an object.")

        column_ref = normalize_sort_column_ref(item.get("column", item.get("label", item.get("name"))))
        direction = str(item.get("direction", direction_default)).strip().lower() or direction_default
        if direction not in {"max", "min"}:
            raise ValueError(f"Sort entry {idx + 1} has invalid direction {direction!r}; use 'max' or 'min'.")

        if column_ref == "":
            result.append(SortSpec("", direction))
        else:
            result.append(SortSpec(labels.resolve(column_ref), direction))

    return result


def sort_controls_to_json_payload(sort_specs: list[SortSpec]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for spec in sort_specs:
        payload.append(
            {
                "column": spec.column if spec.column else NONE_SORT_COLUMN_LABEL,
                "direction": spec.direction,
            }
        )
    return payload


def parse_explorer_state_json(
    text: str,
    labels: LabelManager,
    current_state: dict[str, object],
    n_sort_controls: int,
) -> dict[str, object]:
    state = dict(current_state)
    if text.strip() == "":
        state["filters"] = []
        return state

    payload = json.loads(text)
    if isinstance(payload, list):
        state["filters"] = parse_filters_payload(payload, labels)
        return state
    if not isinstance(payload, dict):
        raise ValueError("Explorer JSON must be an object, or a legacy filter list/dictionary.")

    full_state_keys = {
        "x_axis",
        "y_axis",
        "color_axis",
        "pareto_enabled",
        "draw_all_paretos_for_discretized_colors",
        "draw_all_paretos_for_user",
        "show_best_matching_rows",
        "x_pareto_direction",
        "y_pareto_direction",
        "z_order",
        "x_log_scale",
        "y_log_scale",
        "color_log_scale",
        "best_match_sort",
        "filters",
    }
    if not any(key in payload for key in full_state_keys):
        state["filters"] = parse_filters_payload(payload, labels)
        return state

    if "x_axis" in payload:
        state["x_axis"] = labels.resolve(str(payload["x_axis"]))
    if "y_axis" in payload:
        state["y_axis"] = labels.resolve(str(payload["y_axis"]))
    if "color_axis" in payload:
        state["color_axis"] = labels.resolve(str(payload["color_axis"]))
    if "pareto_enabled" in payload:
        state["pareto_enabled"] = parse_bool_flag(payload["pareto_enabled"], "pareto_enabled")
    if "draw_all_paretos_for_discretized_colors" in payload:
        state["draw_all_paretos_for_discretized_colors"] = parse_bool_flag(
            payload["draw_all_paretos_for_discretized_colors"],
            "draw_all_paretos_for_discretized_colors",
        )
    if "draw_all_paretos_for_user" in payload:
        state["draw_all_paretos_for_user"] = parse_bool_flag(
            payload["draw_all_paretos_for_user"],
            "draw_all_paretos_for_user",
        )
    if "show_best_matching_rows" in payload:
        state["show_best_matching_rows"] = parse_bool_flag(
            payload["show_best_matching_rows"],
            "show_best_matching_rows",
        )
    if "x_pareto_direction" in payload:
        state["x_pareto_direction"] = parse_min_max_direction(payload["x_pareto_direction"], "x_pareto_direction")
    if "y_pareto_direction" in payload:
        state["y_pareto_direction"] = parse_min_max_direction(payload["y_pareto_direction"], "y_pareto_direction")
    if "z_order" in payload:
        z_order = str(payload["z_order"]).strip().lower()
        if z_order not in {"highest", "lowest"}:
            raise ValueError("Field 'z_order' must be 'highest' or 'lowest'.")
        state["z_order"] = z_order
    if "x_log_scale" in payload:
        state["x_log_scale"] = parse_bool_flag(payload["x_log_scale"], "x_log_scale")
    if "y_log_scale" in payload:
        state["y_log_scale"] = parse_bool_flag(payload["y_log_scale"], "y_log_scale")
    if "color_log_scale" in payload:
        state["color_log_scale"] = parse_bool_flag(payload["color_log_scale"], "color_log_scale")
    if "filters" in payload:
        state["filters"] = parse_filters_payload(payload.get("filters"), labels)
    if "best_match_sort" in payload:
        current_sort_specs = current_state.get("best_match_sort", [])
        default_directions = [spec.direction for spec in current_sort_specs if isinstance(spec, SortSpec)]
        state["best_match_sort"] = parse_sort_controls_payload(
            payload.get("best_match_sort"),
            labels,
            n_controls=n_sort_controls,
            default_directions=default_directions,
        )

    return state


def explorer_state_to_json_text(state: dict[str, object]) -> str:
    filters = state.get("filters", [])
    sort_specs = state.get("best_match_sort", [])
    payload = {
        "x_axis": state.get("x_axis", ""),
        "y_axis": state.get("y_axis", ""),
        "color_axis": state.get("color_axis", ""),
        "pareto_enabled": bool(state.get("pareto_enabled", True)),
        "draw_all_paretos_for_discretized_colors": bool(
            state.get("draw_all_paretos_for_discretized_colors", True)
        ),
        "draw_all_paretos_for_user": bool(state.get("draw_all_paretos_for_user", False)),
        "show_best_matching_rows": bool(state.get("show_best_matching_rows", True)),
        "x_pareto_direction": state.get("x_pareto_direction", "max"),
        "y_pareto_direction": state.get("y_pareto_direction", "min"),
        "z_order": state.get("z_order", "highest"),
        "x_log_scale": bool(state.get("x_log_scale", True)),
        "y_log_scale": bool(state.get("y_log_scale", True)),
        "color_log_scale": bool(state.get("color_log_scale", True)),
        "filters": filters_to_json_payload(filters if isinstance(filters, list) else []),
        "best_match_sort": sort_controls_to_json_payload(sort_specs if isinstance(sort_specs, list) else []),
    }
    return json.dumps(payload, indent=2)



def transform_series(values: np.ndarray, log_scale: bool) -> np.ndarray:
    arr = values.astype(np.float32, copy=False)
    if not log_scale:
        return arr
    out = arr.copy()
    np.log2(out, out=out)
    return out


def transform_values_inplace(values: np.ndarray, log_scale: bool) -> np.ndarray:
    out = np.ascontiguousarray(values, dtype=np.float32)
    if log_scale:
        np.log2(out, out=out)
    return out


def transform_xy_indices(
    arrays: dict[str, np.ndarray],
    x_col: str,
    y_col: str,
    indices: np.ndarray,
    x_log: bool,
    y_log: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    if indices.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), 0
    x = arrays[x_col][indices]
    y = arrays[y_col][indices]
    mask = np.isfinite(x) & np.isfinite(y)
    if x_log:
        mask &= x > 0
    if y_log:
        mask &= y > 0
    hidden = int(mask.size - int(mask.sum()))
    if not np.any(mask):
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), hidden
    x_valid = np.ascontiguousarray(x[mask], dtype=np.float32)
    y_valid = np.ascontiguousarray(y[mask], dtype=np.float32)
    if x_log:
        np.log2(x_valid, out=x_valid)
    if y_log:
        np.log2(y_valid, out=y_valid)
    return x_valid, y_valid, hidden


def compute_pareto_frontier(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_direction: str,
    y_direction: str,
    enabled: bool = True,
) -> ParetoResult:
    if x_values.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return ParetoResult(enabled=enabled, relative_indices=empty, display_relative_indices=empty)
    if not enabled:
        all_idx = np.arange(x_values.size, dtype=np.int64)
        display_order = np.lexsort((y_values, x_values))
        return ParetoResult(
            enabled=False,
            relative_indices=all_idx,
            display_relative_indices=display_order.astype(np.int64, copy=False),
        )

    x_pref = x_values if x_direction == "max" else -x_values
    y_pref = y_values if y_direction == "max" else -y_values
    order = np.lexsort((-y_pref, -x_pref))
    x_sorted = x_pref[order]
    y_sorted = y_pref[order]

    group_start = np.empty(order.size, dtype=bool)
    group_start[0] = True
    group_start[1:] = x_sorted[1:] != x_sorted[:-1]
    group_ids = np.cumsum(group_start) - 1
    group_first = np.flatnonzero(group_start)
    group_max_y = y_sorted[group_first]
    prior_best = np.maximum.accumulate(np.concatenate(([np.float32(-np.inf)], group_max_y[:-1])))
    keep_group = group_max_y > prior_best
    keep = keep_group[group_ids] & (y_sorted == group_max_y[group_ids])
    relative_indices = order[keep].astype(np.int64, copy=False)

    if relative_indices.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return ParetoResult(enabled=True, relative_indices=empty, display_relative_indices=empty)

    display_order = np.lexsort((y_values[relative_indices], x_values[relative_indices]))
    display_relative_indices = relative_indices[display_order].astype(np.int64, copy=False)
    return ParetoResult(
        enabled=True,
        relative_indices=relative_indices,
        display_relative_indices=display_relative_indices,
    )


def _resolve_preferred_column(columns: list[str], preferred: str, fallback_index: int = 0) -> str:
    if preferred in columns:
        return preferred
    if not columns:
        return ""
    fallback_index = max(0, min(int(fallback_index), len(columns) - 1))
    return columns[fallback_index]


def compute_grouped_paretos(
    plot_indices: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    group_values: np.ndarray,
    x_direction: str,
    y_direction: str,
) -> list[ParetoSubsetResult]:
    if plot_indices.size == 0:
        return []
    unique_values = np.unique(group_values)
    results: list[ParetoSubsetResult] = []
    for group_value in unique_values:
        subset_positions = np.flatnonzero(group_values == group_value).astype(np.int64, copy=False)
        if subset_positions.size == 0:
            continue
        subset_pareto = compute_pareto_frontier(
            x_values[subset_positions],
            y_values[subset_positions],
            x_direction,
            y_direction,
            enabled=True,
        )
        frontier_plot_positions = subset_positions[subset_pareto.relative_indices].astype(np.int64, copy=False)
        display_plot_positions = subset_positions[subset_pareto.display_relative_indices].astype(np.int64, copy=False)
        frontier_row_indices = plot_indices[frontier_plot_positions].astype(np.int64, copy=False)
        results.append(
            ParetoSubsetResult(
                group_value=float(group_value),
                frontier_plot_positions=frontier_plot_positions,
                display_plot_positions=display_plot_positions,
                frontier_row_indices=frontier_row_indices,
            )
        )
    return results


def sort_best_rows_by_z_order(best_rows: list[BestRowDescriptor], z_order: str) -> list[BestRowDescriptor]:
    reverse = z_order == "lowest"
    return sorted(best_rows, key=lambda item: (item.color_value, item.row_index), reverse=reverse)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive rasterized scatter explorer for very large float-only CSV files.")
    parser.add_argument("csv_path", type=Path, help="Path to the data CSV file.")
    parser.add_argument("--label-csv", type=Path, default="plot_configs/column_index.csv", help="Optional 2-column CSV that maps raw data headers to friendly labels.")
    parser.add_argument(
        "--subgroups-json",
        type=Path,
        default="plot_configs/column_groups.json",
        help=(
            "Optional JSON file that defines named column subgroups for dropdown categories and comparison-table visibility. "
            "Example: {\"subgroups\": [{\"name\": \"Architecture\", \"columns\": [\"banks\", \"ports\"]}], \"always_visible\": [\"id\"]}"
        ),
    )
    parser.add_argument(
        "--simple-axes-csv",
        type=Path,
        default=None,
        help="Deprecated compatibility option; ignored. Use --subgroups-json instead.",
    )
    parser.add_argument(
        "--start-advanced",
        "--advanced",
        action="store_true",
        help="Deprecated compatibility option; ignored because the Advanced toggle has been removed.",
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="In-memory dtype for all columns. float32 is much smaller and usually faster.")
    parser.add_argument("--plot-width", type=int, default=1024, help="Raster width in pixels.")
    parser.add_argument("--plot-height", type=int, default=1024, help="Raster height in pixels.")
    parser.add_argument(
        "--colormap",
        choices=sorted(PALETTE_OPTIONS.keys()),
        default="viridis",
        help="Sequential perceptually uniform colormap for the color overlay.",
    )
    parser.add_argument("--port", type=int, default=5006, help="Local port for the Bokeh server.")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser tab.")
    return parser.parse_args(argv)

def select_best_candidates(
    arrays: dict[str, np.ndarray],
    candidate_idx: np.ndarray,
    sort_specs: list[SortSpec],
) -> tuple[int | None, np.ndarray]:
    if candidate_idx.size == 0:
        return None, np.empty(0, dtype=np.int64)

    candidate_idx = candidate_idx.astype(np.int64, copy=False)
    for sort_spec in sort_specs:
        values = arrays[sort_spec.column][candidate_idx]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        candidate_idx = candidate_idx[finite]
        values = values[finite]
        target = values.max() if sort_spec.direction == "max" else values.min()
        candidate_idx = candidate_idx[values == target]
        if candidate_idx.size <= 1:
            break

    return int(candidate_idx.min()), candidate_idx.astype(np.int64, copy=False)


def find_baseline_info(store: DataStore) -> BaselineInfo:
    if "id" in store.arrays:
        ids = store.arrays["id"]
        matches = np.flatnonzero(np.isfinite(ids) & (ids == 0))
        if matches.size > 0:
            idx = int(matches[0])
            return BaselineInfo(
                index=idx,
                column_title="Baseline (id=0)",
                summary_label=f"id=0 (row index {idx})",
            )
    return BaselineInfo(
        index=0,
        column_title="Baseline (row 0)",
        summary_label="row index 0 fallback (no id=0 row found)",
    )



def make_document_factory(
    store: DataStore,
    labels: LabelManager,
    group_manager: ColumnGroupManager,
    baseline_info: BaselineInfo,
    plot_width: int,
    plot_height: int,
    palette_name: str,
) -> Callable:
    stats_cache = StatsCache(store.arrays)
    filter_mask_cache = FilterMaskCache(store.arrays, store.row_count)
    default_palette = PALETTE_OPTIONS[palette_name]
    default_palette_rgb = _hex_palette_to_rgb(default_palette)
    user_pareto_palette = _resample_palette_segment(Greys256, low_fraction=USER_PARETO_GREYS_LOW_FRACTION, high_fraction=1.0, n_samples=len(default_palette))
    user_pareto_palette_reversed = list(reversed(user_pareto_palette))
    user_pareto_palette_rgb = _hex_palette_to_rgb(user_pareto_palette)
    user_pareto_palette_reversed_rgb = _hex_palette_to_rgb(user_pareto_palette_reversed)
    computer = PlotComputer(store.arrays, width=plot_width, height=plot_height, palette_rgb=default_palette_rgb)
    user_pareto_computer = PlotComputer(store.arrays, width=plot_width, height=plot_height, palette_rgb=user_pareto_palette_rgb)
    user_pareto_computer_reversed = PlotComputer(store.arrays, width=plot_width, height=plot_height, palette_rgb=user_pareto_palette_reversed_rgb)
    grouped_column_options = group_manager.grouped_options(labels)
    grouped_column_options_with_none = group_manager.grouped_options(labels, include_none=True)
    toggleable_groups = group_manager.toggleable_groups()
    always_visible_set = set(group_manager.always_visible_columns)
    right_panel_width = 620
    background_cache: OrderedDict[tuple[str, str, bool, bool], BackgroundCacheEntry] = OrderedDict()
    color_limits_cache: dict[tuple[str, bool], tuple[float, float, int]] = {}
    axis_limits_cache: dict[tuple[str, bool], tuple[float, float, int]] = {}

    def make_document(doc):
        columns = store.columns
        x_default = _resolve_preferred_column(columns, "bw_gbytes", fallback_index=0)
        y_default = _resolve_preferred_column(columns, "metric_e_per_bit_closed", fallback_index=1 if len(columns) > 1 else 0)
        color_default = _resolve_preferred_column(columns, "capacity_gbytes", fallback_index=2 if len(columns) > 2 else 0)

        always_visible_html = ", ".join(labels.label(column) for column in group_manager.always_visible_columns)
        title_bits = [
            f"Loaded <b>{store.row_count:,}</b> rows x <b>{store.n_columns}</b> float columns ",
            #f"as <b>{store.dtype_name}</b> using <b>{store.engine_used}</b> in <b>{store.load_seconds:.2f}s</b>. ",
            f"in <b>{store.load_seconds:.2f}s</b>. ",
            f"Approximate numeric array memory: <b>{human_bytes(store.bytes_in_memory)}</b>. ",
            #f"Column subgroups: <b>{len(group_manager.subgroup_order)}</b>. ",
            f"Baseline comparison row: <b>{baseline_info.summary_label}</b>. ",
        ]
        #if always_visible_html:
        #    title_bits.append(f"Always visible in the comparison table: <b>{always_visible_html}</b>.")
        title = Div(
            text=(
                f"<h2 style='margin:0;'>DreamRAM Explorer</h2>"
                f"<div style='margin-top:4px;'>"
                f"{''.join(title_bits)}"
                f"</div>"
            )
        )

        x_select = Select(title="X axis", value=x_default, options=grouped_column_options, width=220)
        pareto_checkbox = CheckboxGroup(labels=["Use Pareto frontier for display and best-match selection"], active=[0], width=420)
        discrete_pareto_checkbox = CheckboxGroup(
            labels=["Draw all Pareto frontiers for discretized colors"],
            active=[0],
            width=360,
        )
        user_pareto_checkbox = CheckboxGroup(
            labels=[f"Draw all Pareto frontiers for unique {USER_PARETO_COLUMN} values"],
            active=[],
            width=340,
        )
        show_best_match_checkbox = CheckboxGroup(
            labels=["Show best-matching row star markers"],
            active=[0],
            width=280,
        )
        x_pareto_select = Select(
            title="X optimum",
            value="max",
            options=[("max", "Max"), ("min", "Min")],
            width=110,
        )
        y_select = Select(title="Y axis", value=y_default, options=grouped_column_options, width=220)
        y_pareto_select = Select(
            title="Y optimum",
            value="min",
            options=[("max", "Max"), ("min", "Min")],
            width=110,
        )
        color_select = Select(title="Color by", value=color_default, options=grouped_column_options, width=220)
        zorder_select = Select(
            title="Z order when points overlap",
            value="highest",
            options=[("highest", "Highest color on top"), ("lowest", "Lowest color on top")],
            width=220,
        )
        log_checks = CheckboxGroup(labels=["X log2", "Y log2", "Color log2"], active=[LOG_CHECK_X, LOG_CHECK_Y, LOG_CHECK_COLOR])
        update_button = Button(label="Update plot", button_type="primary", width=140)

        add_filter_button = Button(label="Add filter", button_type="default", width=120)
        load_filter_json_button = Button(label="Load JSON", button_type="default", width=140)
        reset_plot_button = Button(label="Reset plot scales", button_type="default", width=150)
        download_scale_select = Select(
            title="PNG scale",
            value="2",
            options=[("1", "1x"), ("2", "2x"), ("3", "3x"), ("4", "4x")],
            width=120,
        )
        download_png_button = Button(label="Download HQ PNG", button_type="success", width=160)

        filter_json = TextAreaInput(
            title="Explorer JSON",
            rows=24,
            width=JSON_PANEL_WIDTH,
            max_length=None,
            placeholder="""{
  "x_axis": "bw_gbytes",
  "pareto_enabled": true,
  "draw_all_paretos_for_discretized_colors": true,
  "draw_all_paretos_for_user": false,
  "show_best_matching_rows": true,
  "x_pareto_direction": "max",
  "y_axis": "metric_e_per_bit_closed",
  "y_pareto_direction": "min",
  "color_axis": "capacity_gbytes",
  "z_order": "highest",
  "x_log_scale": true,
  "y_log_scale": true,
  "color_log_scale": true,
  "filters": [
    {"column": "capacity_gbytes", "min": 16.0, "max": 32.0}
  ],
  "best_match_sort": [
    {"column": "score", "direction": "max"},
    {"column": "(none)", "direction": "max"}
  ]
}""",
        )
        filter_json_help = Div(
            text=(
                "This JSON can store the full explorer state: axes, Pareto directions, the discretized-color Pareto toggle, the user-based Pareto toggle, the best-match star toggle, log flags, z-order, filters, and best-match sorting. Copy this text to save the current explorer configuration, or paste a configuration and press \"Load JSON.\""
            ),
            width=JSON_PANEL_WIDTH,
        )
        filter_json_status = Div(text="", width=JSON_PANEL_WIDTH)
        filters_header = Div(text="<b>Filters</b> (keep rows where min <= value <= max)")
        filters_box = column(sizing_mode="stretch_width")

        sort_controls: list[SortControl] = []
        sort_row_widgets = []
        for idx, default_value in enumerate([columns[0] if columns else "", ""], start=1):
            sort_column = Select(
                title=f"Sort {idx} column",
                value=default_value,
                options=grouped_column_options_with_none,
                width=240,
            )
            sort_direction = Select(
                title=f"Sort {idx} direction",
                value="max",
                options=[("max", "Max"), ("min", "Min")],
                width=120,
            )
            sort_controls.append(SortControl(sort_column, sort_direction))
            sort_row_widgets.extend([sort_column, sort_direction])
        sort_header = Div(text="<b>Best matching row</b> (lexicographic sort, up to 2 levels)")

        empty_image = computer.empty_image
        background_source = ColumnDataSource(data=dict(image=[empty_image], x=[0.0], y=[0.0], dw=[1.0], dh=[1.0]))
        image_source = ColumnDataSource(data=dict(image=[empty_image], x=[0.0], y=[0.0], dw=[1.0], dh=[1.0]))
        anchor_source = ColumnDataSource(data=dict(x=[], y=[]))
        pareto_line_source = ColumnDataSource(data=dict(xs=[], ys=[], line_color=[], line_width=[]))
        pareto_points_source = ColumnDataSource(data=dict(x=[], y=[], c=[], line_color=[], line_width=[]))
        best_source = ColumnDataSource(data=dict(x=[], y=[], c=[], line_color=[]))
        color_mapper = LinearColorMapper(palette=default_palette, low=0.0, high=1.0)

        log_tick_formatter = CustomJSTickFormatter(code=LOG2_TICK_JS)
        linear_tick_formatter = BasicTickFormatter()
        linear_axis_ticker = BasicTicker(desired_num_ticks=9, num_minor_ticks=4)
        linear_color_ticker = BasicTicker(desired_num_ticks=9, num_minor_ticks=0)

        fig = figure(
            width=plot_width,
            height=plot_height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            output_backend="canvas",
            title="",
            x_range=Range1d(0.0, 1.0),
            y_range=Range1d(0.0, 1.0),
            match_aspect=True,
            aspect_scale=1.0,
        )
        fig.grid.visible = False
        fig.toolbar.logo = None
        fig.xaxis.ticker = linear_axis_ticker
        fig.yaxis.ticker = linear_axis_ticker
        fig.xaxis.axis_label_text_font_size = AXIS_LABEL_FONT_SIZE
        fig.yaxis.axis_label_text_font_size = AXIS_LABEL_FONT_SIZE
        fig.xaxis.axis_label_text_font_style = "normal"
        fig.yaxis.axis_label_text_font_style = "normal"
        fig.xaxis.major_label_text_font_size = TICK_LABEL_FONT_SIZE
        fig.yaxis.major_label_text_font_size = TICK_LABEL_FONT_SIZE
        fig.xaxis.major_label_standoff = 14
        fig.yaxis.major_label_standoff = 14
        fig.xaxis.axis_label_standoff = 16
        fig.yaxis.axis_label_standoff = 16
        fig.image_rgba(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=background_source,
            origin="bottom_left",
            anchor="bottom_left",
        )
        fig.image_rgba(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=image_source,
            origin="bottom_left",
            anchor="bottom_left",
        )
        fig.multi_line(
            xs="xs",
            ys="ys",
            source=pareto_line_source,
            line_color={"field": "line_color"},
            line_width={"field": "line_width"},
        )
        fig.scatter(
            x="x",
            y="y",
            source=pareto_points_source,
            marker="circle",
            size=PARETO_POINT_SIZE,
            fill_color={"field": "c", "transform": color_mapper},
            line_color={"field": "line_color"},
            line_width={"field": "line_width"},
        )
        fig.scatter(
            x="x",
            y="y",
            source=best_source,
            marker="star",
            size=BEST_MATCH_POINT_SIZE,
            fill_color={"field": "c", "transform": color_mapper},
            fill_alpha=1.0,
            line_color={"field": "line_color"},
            line_width=2.4,
        )
        fig.scatter(
            x="x",
            y="y",
            source=anchor_source,
            marker="square",
            size=BASELINE_POINT_SIZE,
            fill_color="red",
            line_color="red",
            line_width=2.5,
        )
        color_bar = ColorBar(
            color_mapper=color_mapper,
            title=labels.label(color_default),
            formatter=linear_tick_formatter,
            ticker=linear_color_ticker,
            width=45,
            title_standoff=16,
            label_standoff=14,
            title_text_font_size=COLOR_BAR_TITLE_FONT_SIZE,
            title_text_font_style="normal",
            major_label_text_font_size=COLOR_BAR_TICK_FONT_SIZE,
        )
        fig.min_border_right = 150
        fig.min_border_left = 150
        fig.min_border_bottom = 150
        fig.min_border_top = 150
        fig.add_layout(color_bar, "right")
        fig.xaxis.axis_label = labels.label(x_default)
        fig.yaxis.axis_label = labels.label(y_default)

        download_png_button.js_on_click(
            CustomJS(
                args=dict(plot=fig, scale_select=download_scale_select),
                code=r"""
const scale = Math.max(1, Number(scale_select.value) || 1)
const view = Bokeh.index[plot.id]
if (view == null) {
  console.error("Could not find plot view for PNG export.")
  return
}

const exported = view.export("png", true)
const source_canvas = exported.canvas
const out = document.createElement("canvas")
out.width = Math.max(1, Math.round(source_canvas.width * scale))
out.height = Math.max(1, Math.round(source_canvas.height * scale))

const ctx = out.getContext("2d")
if (ctx == null) {
  console.error("Could not create PNG export canvas.")
  return
}
ctx.imageSmoothingEnabled = true
ctx.drawImage(source_canvas, 0, 0, out.width, out.height)

const sanitize = (text) => String(text)
  .replace(/[^A-Za-z0-9_\-]+/g, "_")
  .replace(/^_+|_+$/g, "")

const x_label = sanitize(plot.xaxis[0]?.axis_label || "x") || "x"
const y_label = sanitize(plot.yaxis[0]?.axis_label || "y") || "y"
const filename = `${x_label}_vs_${y_label}_${scale}x.png`

out.toBlob((blob) => {
  if (blob == null) {
    console.error("PNG export failed.")
    return
  }
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.target = "_blank"
  link.dispatchEvent(new MouseEvent("click"))
  setTimeout(() => URL.revokeObjectURL(url), 1500)
}, "image/png")
""",
            )
        )

        best_row_source = ColumnDataSource(data=dict(field=[], baseline_value=[]))
        best_row_table = DataTable(
            source=best_row_source,
            columns=[
                TableColumn(field="field", title="Column"),
                TableColumn(field="baseline_value", title=baseline_info.column_title),
            ],
            width=right_panel_width,
            height=plot_height,
            index_position=None,
            autosize_mode="force_fit",
            sortable=False,
            reorderable=False,
            editable=False,
        )
        best_row_summary = Div(text="Press <b>Update plot</b> to compute the best matching row.", width=right_panel_width)
        subgroup_visibility_header = Div(text="<b>Show/Hide subgroup columns:</b>", width=right_panel_width)
        subgroup_visibility_checks = CheckboxGroup(labels=toggleable_groups, active=list(range(len(toggleable_groups))))
        subgroup_visibility_checks.visible = bool(toggleable_groups)
        subgroup_visibility_header.visible = bool(toggleable_groups)
        subgroup_visibility_note_parts = [
            "NOTE: for the table to update, you may need to toggle a subgroup option above."
        ]
        #if always_visible_html:
        #    subgroup_visibility_note_parts.append(f"Always visible: <b>{always_visible_html}</b>.")
        subgroup_visibility_note = Div(text=" ".join(subgroup_visibility_note_parts), width=right_panel_width)
        status = Div(text="Press <b>Update plot</b> to render.", width=plot_width + right_panel_width)

        filter_rows: list[FilterRow] = []
        last_best_rows: list[BestRowDescriptor] = []
        last_filter_match_count = 0
        last_selection_count = 0
        last_sort_specs: list[SortSpec] = []
        last_tied_count = 0
        last_grouped_pareto_column: str | None = None
        current_reset_bounds: PlotBounds | None = None
        current_reset_color_low = 0.0
        current_reset_color_high = 1.0

        def update_filter_info(fr: FilterRow) -> None:
            lo, hi = stats_cache.get(fr.column_select.value)
            fr.info.text = f"Range: {lo:.6g} to {hi:.6g}"

        def create_filter_row(initial_column: str | None = None, lo: float | None = None, hi: float | None = None) -> FilterRow:
            column_name = initial_column or columns[0]
            col_select = Select(title="Column", value=column_name, options=grouped_column_options, width=260)
            min_input = NumericInput(title="Min", mode="float", value=lo, width=150)
            max_input = NumericInput(title="Max", mode="float", value=hi, width=150)
            remove_button = Button(label="Remove", button_type="danger", width=85)
            info = Div(width=220)
            container = row(col_select, min_input, max_input, remove_button, info)
            fr = FilterRow(col_select, min_input, max_input, remove_button, info, container)
            col_select.on_change("value", lambda attr, old, new, fr=fr: update_filter_info(fr))
            remove_button.on_click(lambda fr=fr: remove_filter_row(fr))
            update_filter_info(fr)
            return fr

        def refresh_filter_widgets() -> None:
            filters_box.children = [fr.container for fr in filter_rows]
            for fr in filter_rows:
                fr.column_select.options = grouped_column_options
                update_filter_info(fr)

        def add_filter_row(initial_column: str | None = None, lo: float | None = None, hi: float | None = None) -> None:
            preferred_column = initial_column or columns[0]
            fr = create_filter_row(preferred_column, lo=lo, hi=hi)
            filter_rows.append(fr)
            refresh_filter_widgets()

        def remove_filter_row(fr: FilterRow) -> None:
            if fr in filter_rows:
                filter_rows.remove(fr)
            refresh_filter_widgets()

        def replace_filters(filters: list[tuple[str, float | None, float | None]]) -> None:
            filter_rows.clear()
            for column_name, lo, hi in filters:
                fr = create_filter_row(column_name, lo=lo, hi=hi)
                filter_rows.append(fr)
            refresh_filter_widgets()

        def collect_filters() -> list[tuple[str, float | None, float | None]]:
            result: list[tuple[str, float | None, float | None]] = []
            for fr in filter_rows:
                column_name = fr.column_select.value
                lo = None if fr.min_input.value in (None, "") else float(fr.min_input.value)
                hi = None if fr.max_input.value in (None, "") else float(fr.max_input.value)
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError(f"Filter for {labels.label(column_name)} has min > max.")
                if lo is None and hi is None:
                    continue
                result.append((column_name, lo, hi))
            return result

        def collect_all_sort_controls() -> list[SortSpec]:
            return [SortSpec(control.column_select.value, control.direction_select.value) for control in sort_controls]

        def current_ui_state(filters: list[tuple[str, float | None, float | None]] | None = None) -> dict[str, object]:
            active_logs = set(log_checks.active)
            return {
                "x_axis": x_select.value,
                "pareto_enabled": 0 in pareto_checkbox.active,
                "draw_all_paretos_for_discretized_colors": 0 in discrete_pareto_checkbox.active,
                "draw_all_paretos_for_user": 0 in user_pareto_checkbox.active,
                "show_best_matching_rows": 0 in show_best_match_checkbox.active,
                "x_pareto_direction": x_pareto_select.value,
                "y_axis": y_select.value,
                "y_pareto_direction": y_pareto_select.value,
                "color_axis": color_select.value,
                "z_order": zorder_select.value,
                "x_log_scale": LOG_CHECK_X in active_logs,
                "y_log_scale": LOG_CHECK_Y in active_logs,
                "color_log_scale": LOG_CHECK_COLOR in active_logs,
                "filters": list(filters) if filters is not None else collect_filters(),
                "best_match_sort": collect_all_sort_controls(),
            }

        def sync_json_from_ui(filters: list[tuple[str, float | None, float | None]] | None = None) -> None:
            filter_json.value = explorer_state_to_json_text(current_ui_state(filters=filters))

        def apply_ui_state(state: dict[str, object]) -> None:
            x_select.value = str(state.get("x_axis", x_select.value))
            pareto_checkbox.active = [0] if bool(state.get("pareto_enabled", True)) else []
            discrete_pareto_checkbox.active = [0] if bool(state.get("draw_all_paretos_for_discretized_colors", True)) else []
            user_pareto_checkbox.active = [0] if bool(state.get("draw_all_paretos_for_user", False)) else []
            show_best_match_checkbox.active = [0] if bool(state.get("show_best_matching_rows", True)) else []
            x_pareto_select.value = str(state.get("x_pareto_direction", x_pareto_select.value))
            y_select.value = str(state.get("y_axis", y_select.value))
            y_pareto_select.value = str(state.get("y_pareto_direction", y_pareto_select.value))
            color_select.value = str(state.get("color_axis", color_select.value))
            zorder_select.value = str(state.get("z_order", zorder_select.value))
            new_active: list[int] = []
            if bool(state.get("x_log_scale", False)):
                new_active.append(LOG_CHECK_X)
            if bool(state.get("y_log_scale", False)):
                new_active.append(LOG_CHECK_Y)
            if bool(state.get("color_log_scale", False)):
                new_active.append(LOG_CHECK_COLOR)
            log_checks.active = new_active
            filters = state.get("filters", [])
            replace_filters(filters if isinstance(filters, list) else [])
            sort_state = state.get("best_match_sort", [])
            if isinstance(sort_state, list):
                for idx, control in enumerate(sort_controls):
                    if idx < len(sort_state) and isinstance(sort_state[idx], SortSpec):
                        control.column_select.value = sort_state[idx].column
                        control.direction_select.value = sort_state[idx].direction
                    else:
                        control.column_select.value = ""
                        control.direction_select.value = "max"

        def load_filter_json_clicked() -> None:
            try:
                try:
                    current_filters = collect_filters()
                except Exception:
                    current_filters = []
                state = parse_explorer_state_json(
                    filter_json.value,
                    labels,
                    current_state=current_ui_state(filters=current_filters),
                    n_sort_controls=len(sort_controls),
                )
            except Exception as exc:
                filter_json_status.text = f"<span style='color:#b00020;'><b>Explorer JSON error:</b> {exc}</span>"
                return
            apply_ui_state(state)
            sync_json_from_ui(filters=state.get("filters") if isinstance(state.get("filters"), list) else [])
            filter_json_status.text = "Loaded explorer state from JSON."
            do_update()

        load_filter_json_button.on_click(load_filter_json_clicked)
        add_filter_button.on_click(lambda: add_filter_row())

        def collect_sort_specs() -> list[SortSpec]:
            return [spec for spec in collect_all_sort_controls() if spec.column != ""]

        def sort_summary_html(sort_specs: list[SortSpec]) -> str:
            if not sort_specs:
                return "(row-index tie-break only)"
            parts = [f"{labels.label(spec.column)} ({spec.direction})" for spec in sort_specs]
            return " &rarr; ".join(parts)

        def visible_toggleable_groups() -> set[str]:
            visible: set[str] = set()
            for idx in subgroup_visibility_checks.active:
                if 0 <= idx < len(toggleable_groups):
                    visible.add(toggleable_groups[idx])
            return visible

        def format_table_value(column_name: str, row_index: int | None) -> str:
            if row_index is None:
                return ""
            if column_name == "__row_index__":
                return str(int(row_index))
            return format_best_row_value(column_name, float(store.arrays[column_name][row_index]))

        def best_row_column_title(best_row: BestRowDescriptor, is_multi: bool) -> str:
            if not is_multi:
                return "Best match"
            if best_row.group_column is not None and best_row.group_value is not None:
                group_value_text = format_best_row_value(best_row.group_column, best_row.group_value)
                return f"Best @ {labels.label(best_row.group_column)}={group_value_text}"
            color_value_text = format_best_row_value(color_select.value, best_row.color_value)
            return f"Best @ {labels.label(color_select.value)}={color_value_text}"

        def build_comparison_table_data(best_rows: list[BestRowDescriptor]) -> tuple[dict[str, list[str]], list[TableColumn]]:
            fields: list[str] = ["Row index"]
            baseline_values: list[str] = [format_table_value("__row_index__", baseline_info.index)]
            visible_groups = visible_toggleable_groups()
            ordered_best_rows = sort_best_rows_by_z_order(list(best_rows), zorder_select.value)
            data: dict[str, list[str]] = {
                "field": fields,
                "baseline_value": baseline_values,
            }
            table_columns: list[TableColumn] = [
                TableColumn(field="field", title="Column"),
                TableColumn(field="baseline_value", title=baseline_info.column_title),
            ]
            is_multi = len(ordered_best_rows) > 1
            best_column_keys = [f"best_value_{idx}" for idx in range(len(ordered_best_rows))]
            for key, best_row in zip(best_column_keys, ordered_best_rows):
                data[key] = [format_table_value("__row_index__", best_row.row_index)]
                table_columns.append(TableColumn(field=key, title=best_row_column_title(best_row, is_multi)))

            for column_name in group_manager.always_visible_columns:
                fields.append(labels.label(column_name))
                baseline_values.append(format_table_value(column_name, baseline_info.index))
                for key, best_row in zip(best_column_keys, ordered_best_rows):
                    data[key].append(format_table_value(column_name, best_row.row_index))
            for group_name in group_manager.subgroup_order:
                if group_name not in visible_groups:
                    continue
                for column_name in group_manager.subgroup_to_columns.get(group_name, []):
                    if column_name in always_visible_set:
                        continue
                    fields.append(labels.label(column_name))
                    baseline_values.append(format_table_value(column_name, baseline_info.index))
                    for key, best_row in zip(best_column_keys, ordered_best_rows):
                        data[key].append(format_table_value(column_name, best_row.row_index))
            return data, table_columns

        def update_best_row_display(
            best_rows: list[BestRowDescriptor],
            filter_match_count: int,
            selection_count: int,
            sort_specs: list[SortSpec],
            tied_count: int,
            grouped_pareto_column: str | None,
        ) -> None:
            nonlocal last_best_rows, last_filter_match_count, last_selection_count, last_sort_specs, last_tied_count, last_grouped_pareto_column
            last_best_rows = list(best_rows)
            last_filter_match_count = filter_match_count
            last_selection_count = selection_count
            last_sort_specs = list(sort_specs)
            last_tied_count = tied_count
            last_grouped_pareto_column = grouped_pareto_column
            table_data, table_columns = build_comparison_table_data(best_rows)
            best_row_source.data = table_data
            best_row_table.columns = table_columns
            if grouped_pareto_column is not None:
                selection_name = f"per-{labels.label(grouped_pareto_column)} Pareto frontiers"
                mode_name = f"per-{labels.label(grouped_pareto_column)} Pareto"
            elif 0 in pareto_checkbox.active:
                selection_name = "x/y Pareto frontier"
                mode_name = "Pareto"
            else:
                selection_name = "filtered plottable set"
                mode_name = "plottable"
            if not best_rows:
                best_row_summary.text = (
                    f"No filtered plottable rows remain in the current {selection_name}. "
                    f"Rows satisfying filters: <b>{filter_match_count:,}</b>. "
                    f"Selection-set points used for best-row selection: <b>{selection_count:,}</b>. "
                    f"Baseline reference: <b>{baseline_info.summary_label}</b>. "
                    f"Sort order: <b>{sort_summary_html(sort_specs)}</b>."
                )
                return
            best_row_summary.text = (
                f"Best filtered {mode_name} row marker(s): <b>{len(best_rows):,}</b>. "
                f"Rows satisfying filters: <b>{filter_match_count:,}</b>. "
                f"Selection-set points used for best-row selection: <b>{selection_count:,}</b>. "
                f"Additional ties after sorts across the selection sets: <b>{tied_count:,}</b>. "
                f"Baseline reference: <b>{baseline_info.summary_label}</b>. "
                f"Sort order: <b>{sort_summary_html(sort_specs)}</b>."
            )

        def refresh_best_row_visibility_only() -> None:
            update_best_row_display(
                last_best_rows,
                last_filter_match_count,
                last_selection_count,
                last_sort_specs,
                last_tied_count,
                last_grouped_pareto_column,
            )

        def apply_axis_formatters(x_log: bool, y_log: bool, color_log: bool, x_col: str, y_col: str, color_col: str) -> None:
            fig.xaxis.formatter = log_tick_formatter if x_log else linear_tick_formatter
            fig.yaxis.formatter = log_tick_formatter if y_log else linear_tick_formatter
            fig.xaxis.axis_label = labels.label(x_col)
            fig.yaxis.axis_label = labels.label(y_col)
            color_bar.title = labels.label(color_col)
            color_bar.formatter = log_tick_formatter if color_log else linear_tick_formatter

        def update_axis_tickers(
            x_log: bool,
            y_log: bool,
            x_range: tuple[float, float],
            y_range: tuple[float, float],
        ) -> None:
            fig.xaxis.ticker = make_log_axis_ticker(*x_range) if x_log else linear_axis_ticker
            fig.yaxis.ticker = make_log_axis_ticker(*y_range) if y_log else linear_axis_ticker

        def update_color_bar_ticks(color_log: bool, color_tick_low: float, color_tick_high: float) -> None:
            if not color_log:
                ticks = build_linear_power_tick_values(color_tick_low, color_tick_high)
                if ticks:
                    color_bar.ticker = FixedTicker(ticks=ticks, minor_ticks=[], num_minor_ticks=0)
                else:
                    color_bar.ticker = linear_color_ticker
                color_bar.formatter = linear_tick_formatter
                color_bar.major_label_overrides = {}
                return
            ticks = build_log_tick_values(color_tick_low, color_tick_high)
            color_bar.ticker = FixedTicker(ticks=ticks, minor_ticks=[], num_minor_ticks=0)
            color_bar.formatter = linear_tick_formatter
            color_bar.major_label_overrides = {
                float(tick): format_log_tick_value(float(2.0**tick))
                for tick in ticks
            }

        def get_background_entry(x_col: str, y_col: str, x_log: bool, y_log: bool) -> BackgroundCacheEntry:
            key = (x_col, y_col, x_log, y_log)
            cached = background_cache.get(key)
            if cached is not None:
                background_cache.move_to_end(key)
                return cached
            x_arr = store.arrays[x_col]
            y_arr = store.arrays[y_col]
            full_xy_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if x_log:
                full_xy_mask &= x_arr > 0
            if y_log:
                full_xy_mask &= y_arr > 0
            if not np.any(full_xy_mask):
                raise EmptySelectionError(
                    "No plottable rows exist in the full dataset for the selected x/y axes and log-scale settings."
                )
            entry = computer.render_background_from_mask(x_arr, y_arr, full_xy_mask, x_log, y_log)
            background_cache[key] = entry
            background_cache.move_to_end(key)
            while len(background_cache) > MAX_BACKGROUND_CACHE_ENTRIES:
                background_cache.popitem(last=False)
            return entry

        def get_full_axis_limits(axis_col: str, axis_log: bool) -> tuple[float, float, int]:
            key = (axis_col, axis_log)
            cached = axis_limits_cache.get(key)
            if cached is not None:
                return cached
            arr = store.arrays[axis_col]
            finite_mask = np.isfinite(arr)
            valid_count = int(np.count_nonzero(finite_mask))
            if valid_count == 0:
                result = (0.0, 1.0, 0)
            elif axis_log:
                valid_mask = finite_mask & (arr > 0)
                valid_count = int(np.count_nonzero(valid_mask))
                if valid_count == 0:
                    result = (0.0, 1.0, 0)
                else:
                    raw_low = float(np.min(arr, where=valid_mask, initial=np.inf))
                    raw_high = float(np.max(arr, where=valid_mask, initial=-np.inf))
                    low = float(np.log2(raw_low))
                    high = float(np.log2(raw_high))
                    low, high = _ensure_nonzero_extent(low, high)
                    result = (*_padded_range(low, high, frac=DEFAULT_MARGIN_FRAC), valid_count)
            else:
                low = float(np.min(arr, where=finite_mask, initial=np.inf))
                high = float(np.max(arr, where=finite_mask, initial=-np.inf))
                low, high = _ensure_nonzero_extent(low, high)
                result = (*_padded_range(low, high, frac=DEFAULT_MARGIN_FRAC), valid_count)
            axis_limits_cache[key] = result
            return result

        def get_full_color_limits(color_col: str, color_log: bool) -> tuple[float, float, float, float, int]:
            key = (color_col, color_log)
            cached = color_limits_cache.get(key)
            if cached is not None:
                return cached
            arr = store.arrays[color_col]
            finite_mask = np.isfinite(arr)
            valid_count = int(np.count_nonzero(finite_mask))
            if valid_count == 0:
                result = (0.0, 1.0, 0.0, 1.0, 0)
            elif color_log:
                valid_mask = finite_mask & (arr > 0)
                valid_count = int(np.count_nonzero(valid_mask))
                if valid_count == 0:
                    result = (0.0, 1.0, 0.0, 1.0, 0)
                else:
                    raw_low = float(np.min(arr, where=valid_mask, initial=np.inf))
                    raw_high = float(np.max(arr, where=valid_mask, initial=-np.inf))
                    tick_low = float(np.log2(raw_low))
                    tick_high = float(np.log2(raw_high))
                    display_low, display_high = expand_log_display_range_for_power_ticks(tick_low, tick_high)
                    result = (display_low, display_high, tick_low, tick_high, valid_count)
            else:
                raw_low = float(np.min(arr, where=finite_mask, initial=np.inf))
                raw_high = float(np.max(arr, where=finite_mask, initial=-np.inf))
                display_low, display_high = expand_linear_display_range_for_power_ticks(raw_low, raw_high)
                result = (display_low, display_high, raw_low, raw_high, valid_count)
            color_limits_cache[key] = result
            return result

        def apply_plot_scales(bounds: PlotBounds, color_low: float, color_high: float) -> None:
            fig.x_range.start, fig.x_range.end = bounds.x_range
            fig.y_range.start, fig.y_range.end = bounds.y_range
            fig.x_range.reset_start, fig.x_range.reset_end = bounds.x_range
            fig.y_range.reset_start, fig.y_range.reset_end = bounds.y_range
            color_mapper.low = float(color_low)
            color_mapper.high = float(color_high)

        def reset_plot_view() -> None:
            nonlocal current_reset_bounds, current_reset_color_low, current_reset_color_high
            if current_reset_bounds is None:
                return
            apply_plot_scales(current_reset_bounds, current_reset_color_low, current_reset_color_high)

        def do_update() -> None:
            nonlocal current_reset_bounds, current_reset_color_low, current_reset_color_high
            t0 = time.perf_counter()
            x_col = x_select.value
            pareto_enabled = 0 in pareto_checkbox.active
            discrete_pareto_requested = 0 in discrete_pareto_checkbox.active
            show_best_match_rows = 0 in show_best_match_checkbox.active
            x_pareto_direction = x_pareto_select.value
            y_col = y_select.value
            y_pareto_direction = y_pareto_select.value
            color_col = color_select.value
            x_log = LOG_CHECK_X in log_checks.active
            y_log = LOG_CHECK_Y in log_checks.active
            color_log = LOG_CHECK_COLOR in log_checks.active
            sort_specs = collect_sort_specs()
            try:
                filters = collect_filters()
            except Exception as exc:
                status.text = f"<span style='color:#b00020;'><b>Invalid filters:</b> {exc}</span>"
                return
            sync_json_from_ui(filters=filters)
            apply_axis_formatters(x_log, y_log, color_log, x_col, y_col, color_col)
            try:
                background_entry = get_background_entry(x_col, y_col, x_log, y_log)
            except EmptySelectionError as exc:
                background_source.data = dict(image=[empty_image], x=[0.0], y=[0.0], dw=[1.0], dh=[1.0])
                image_source.data = dict(image=[empty_image], x=[0.0], y=[0.0], dw=[1.0], dh=[1.0])
                anchor_source.data = dict(x=[], y=[])
                pareto_line_source.data = dict(xs=[], ys=[], line_color=[], line_width=[])
                pareto_points_source.data = dict(x=[], y=[], c=[], line_color=[], line_width=[])
                best_source.data = dict(x=[], y=[], c=[], line_color=[])
                fallback_bounds = PlotBounds(0.0, 1.0, 0.0, 1.0, (0.0, 1.0), (0.0, 1.0))
                apply_plot_scales(fallback_bounds, 0.0, 1.0)
                update_axis_tickers(x_log, y_log, fallback_bounds.x_range, fallback_bounds.y_range)
                update_color_bar_ticks(color_log, 0.0, 1.0)
                update_best_row_display([], 0, 0, sort_specs, 0, False)
                elapsed = time.perf_counter() - t0
                status.text = f"{exc} Elapsed: <b>{elapsed:.3f}s</b>."
                return
            full_color_low, full_color_high, color_tick_low, color_tick_high, full_color_count = get_full_color_limits(color_col, color_log)
            x_axis_low, x_axis_high, _ = get_full_axis_limits(x_col, x_log)
            y_axis_low, y_axis_high, _ = get_full_axis_limits(y_col, y_log)
            current_reset_bounds = PlotBounds(
                x_img_min=background_entry.bounds.x_img_min,
                x_img_max=background_entry.bounds.x_img_max,
                y_img_min=background_entry.bounds.y_img_min,
                y_img_max=background_entry.bounds.y_img_max,
                x_range=(x_axis_low, x_axis_high),
                y_range=(y_axis_low, y_axis_high),
            )
            current_reset_color_low = full_color_low
            current_reset_color_high = full_color_high
            apply_plot_scales(current_reset_bounds, full_color_low, full_color_high)
            update_axis_tickers(x_log, y_log, current_reset_bounds.x_range, current_reset_bounds.y_range)
            update_color_bar_ticks(color_log, color_tick_low, color_tick_high)

            filter_mask = filter_mask_cache.get(filters)
            filter_match_count = int(filter_mask.sum())
            baseline_indices = np.array([baseline_info.index], dtype=np.int64) if store.row_count > 0 else np.empty(0, dtype=np.int64)
            baseline_x, baseline_y, baseline_hidden = transform_xy_indices(store.arrays, x_col, y_col, baseline_indices, x_log, y_log)
            anchor_source.data = dict(x=baseline_x, y=baseline_y)

            plot_mask = computer.build_plot_mask(filter_mask, x_col, y_col, color_col, x_log, y_log, color_log)
            if np.any(plot_mask):
                plot_indices = np.flatnonzero(plot_mask).astype(np.int64, copy=False)
                x_plot = transform_values_inplace(store.arrays[x_col][plot_indices], x_log)
                y_plot = transform_values_inplace(store.arrays[y_col][plot_indices], y_log)
            else:
                plot_indices = np.empty(0, dtype=np.int64)
                x_plot = np.empty(0, dtype=np.float32)
                y_plot = np.empty(0, dtype=np.float32)
            use_discrete_paretos = False
            use_user_paretos = False
            unique_color_count = 0
            user_hidden_count = 0
            grouped_pareto_column: str | None = None
            discrete_pareto_results: list[ParetoSubsetResult] = []
            user_pareto_results: list[ParetoSubsetResult] = []
            if plot_indices.size > 0:
                plot_color_raw = np.ascontiguousarray(store.arrays[color_col][plot_indices])
            else:
                plot_color_raw = np.empty(0, dtype=np.float32)

            user_pareto_requested = 0 in user_pareto_checkbox.active
            if pareto_enabled and user_pareto_requested and USER_PARETO_COLUMN in store.arrays and plot_indices.size > 0:
                plot_user_raw = np.ascontiguousarray(store.arrays[USER_PARETO_COLUMN][plot_indices])
                finite_user_mask = np.isfinite(plot_user_raw)
                user_hidden_count = int(plot_user_raw.size - int(np.count_nonzero(finite_user_mask)))
                if np.any(finite_user_mask):
                    user_pareto_results = compute_grouped_paretos(
                        plot_indices[finite_user_mask],
                        x_plot[finite_user_mask],
                        y_plot[finite_user_mask],
                        plot_user_raw[finite_user_mask],
                        x_pareto_direction,
                        y_pareto_direction,
                    )
                    use_user_paretos = True
                    grouped_pareto_column = USER_PARETO_COLUMN
            elif pareto_enabled and discrete_pareto_requested and plot_indices.size > 0:
                unique_color_count = int(np.unique(plot_color_raw).size)
                if unique_color_count <= 7:
                    discrete_pareto_results = compute_grouped_paretos(
                        plot_indices,
                        x_plot,
                        y_plot,
                        plot_color_raw,
                        x_pareto_direction,
                        y_pareto_direction,
                    )
                    use_discrete_paretos = True
                    grouped_pareto_column = color_col

            if use_user_paretos:
                render_indices = plot_indices
                if zorder_select.value == "highest":
                    active_overlay_computer = user_pareto_computer_reversed
                    active_palette = user_pareto_palette_reversed
                else:
                    active_overlay_computer = user_pareto_computer
                    active_palette = user_pareto_palette
                background_image = empty_image
            else:
                render_indices = plot_indices
                active_overlay_computer = computer
                active_palette = default_palette
                background_image = background_entry.image

            color_mapper.palette = active_palette
            background_source.data = dict(
                image=[background_image],
                x=[background_entry.bounds.x_img_min],
                y=[background_entry.bounds.y_img_min],
                dw=[background_entry.bounds.x_img_max - background_entry.bounds.x_img_min],
                dh=[background_entry.bounds.y_img_max - background_entry.bounds.y_img_min],
            )
            render = active_overlay_computer.render_color_overlay_from_indices(
                x_values=store.arrays[x_col],
                y_values=store.arrays[y_col],
                c_values=store.arrays[color_col],
                indices=render_indices,
                x_log=x_log,
                y_log=y_log,
                color_log=color_log,
                z_order=zorder_select.value,
                bounds=background_entry.bounds,
                color_low=full_color_low,
                color_high=full_color_high,
            )
            image_source.data = dict(
                image=[render["image"]],
                x=[background_entry.bounds.x_img_min],
                y=[background_entry.bounds.y_img_min],
                dw=[background_entry.bounds.x_img_max - background_entry.bounds.x_img_min],
                dh=[background_entry.bounds.y_img_max - background_entry.bounds.y_img_min],
            )

            best_rows: list[BestRowDescriptor] = []
            selection_count = 0
            tied_count = 0
            pareto_line_xs: list[list[float]] = []
            pareto_line_ys: list[list[float]] = []
            pareto_line_colors: list[str] = []
            pareto_line_widths: list[float] = []
            pareto_points_x: list[float] = []
            pareto_points_y: list[float] = []
            pareto_points_c: list[float] = []
            pareto_point_line_colors: list[str] = []
            pareto_point_line_widths: list[float] = []
            best_x: list[float] = []
            best_y: list[float] = []
            best_c: list[float] = []
            best_line_colors: list[str] = []

            if pareto_enabled:
                if use_user_paretos:
                    user_plot_index_lookup = {int(row_index): plot_pos for plot_pos, row_index in enumerate(plot_indices.tolist())}
                    for subset in reversed(user_pareto_results):
                        selection_count += int(subset.frontier_row_indices.size)
                        if subset.display_plot_positions.size > 0:
                            pareto_line_xs.append(x_plot[subset.display_plot_positions].tolist())
                            pareto_line_ys.append(y_plot[subset.display_plot_positions].tolist())
                            user_line_color = user_pareto_line_color(subset.group_value)
                            pareto_line_colors.append(user_line_color)
                            pareto_line_widths.append(USER_PARETO_LINE_WIDTH)
                            subset_c = transform_values_inplace(
                                store.arrays[color_col][plot_indices[subset.display_plot_positions]],
                                color_log,
                            )
                            subset_point_count = int(subset.display_plot_positions.size)
                            pareto_points_x.extend(x_plot[subset.display_plot_positions].tolist())
                            pareto_points_y.extend(y_plot[subset.display_plot_positions].tolist())
                            pareto_points_c.extend(subset_c.tolist())
                            pareto_point_line_colors.extend([user_line_color] * subset_point_count)
                            pareto_point_line_widths.extend([USER_PARETO_POINT_LINE_WIDTH] * subset_point_count)
                        best_index, tied_indices = select_best_candidates(store.arrays, subset.frontier_row_indices, sort_specs)
                        tied_count += int(max(tied_indices.size - 1, 0))
                        if best_index is None:
                            continue
                        plot_pos = user_plot_index_lookup.get(int(best_index))
                        if plot_pos is None:
                            continue
                        best_rows.append(
                            BestRowDescriptor(
                                row_index=best_index,
                                color_value=float(store.arrays[color_col][best_index]),
                                group_column=USER_PARETO_COLUMN,
                                group_value=subset.group_value,
                            )
                        )
                        best_x.append(float(x_plot[plot_pos]))
                        best_y.append(float(y_plot[plot_pos]))
                        best_c.extend(
                            transform_values_inplace(
                                store.arrays[color_col][np.array([best_index], dtype=np.int64)],
                                color_log,
                            ).tolist()
                        )
                        best_line_colors.append(user_pareto_line_color(subset.group_value))
                elif use_discrete_paretos:
                    plot_index_lookup = {int(row_index): plot_pos for plot_pos, row_index in enumerate(plot_indices.tolist())}
                    for subset in reversed(discrete_pareto_results):
                        selection_count += int(subset.frontier_row_indices.size)
                        if subset.display_plot_positions.size > 0:
                            pareto_line_xs.append(x_plot[subset.display_plot_positions].tolist())
                            pareto_line_ys.append(y_plot[subset.display_plot_positions].tolist())
                            pareto_line_colors.append("black")
                            pareto_line_widths.append(2.0)
                            subset_c = transform_values_inplace(
                                store.arrays[color_col][plot_indices[subset.display_plot_positions]],
                                color_log,
                            )
                            subset_point_count = int(subset.display_plot_positions.size)
                            pareto_points_x.extend(x_plot[subset.display_plot_positions].tolist())
                            pareto_points_y.extend(y_plot[subset.display_plot_positions].tolist())
                            pareto_points_c.extend(subset_c.tolist())
                            pareto_point_line_colors.extend(["black"] * subset_point_count)
                            pareto_point_line_widths.extend([0.7] * subset_point_count)
                        best_index, tied_indices = select_best_candidates(store.arrays, subset.frontier_row_indices, sort_specs)
                        tied_count += int(max(tied_indices.size - 1, 0))
                        if best_index is None:
                            continue
                        plot_pos = plot_index_lookup.get(int(best_index))
                        if plot_pos is None:
                            continue
                        best_rows.append(
                            BestRowDescriptor(
                                row_index=best_index,
                                color_value=float(store.arrays[color_col][best_index]),
                                group_column=color_col,
                                group_value=subset.group_value,
                            )
                        )
                        best_x.append(float(x_plot[plot_pos]))
                        best_y.append(float(y_plot[plot_pos]))
                        best_c.extend(
                            transform_values_inplace(
                                store.arrays[color_col][np.array([best_index], dtype=np.int64)],
                                color_log,
                            ).tolist()
                        )
                        best_line_colors.append("red")
                else:
                    pareto_result = compute_pareto_frontier(
                        x_plot,
                        y_plot,
                        x_pareto_direction,
                        y_pareto_direction,
                        enabled=True,
                    )
                    pareto_positions = pareto_result.relative_indices
                    pareto_display_positions = pareto_result.display_relative_indices
                    candidate_row_indices = (
                        plot_indices[pareto_positions] if pareto_positions.size > 0 else np.empty(0, dtype=np.int64)
                    )
                    selection_count = int(candidate_row_indices.size)
                    best_index, tied_indices = select_best_candidates(store.arrays, candidate_row_indices, sort_specs)
                    tied_count = int(max(tied_indices.size - 1, 0))
                    if pareto_display_positions.size > 0:
                        pareto_x = x_plot[pareto_display_positions]
                        pareto_y = y_plot[pareto_display_positions]
                        pareto_c = transform_values_inplace(store.arrays[color_col][plot_indices[pareto_display_positions]], color_log)
                        pareto_line_xs = [pareto_x.tolist()]
                        pareto_line_ys = [pareto_y.tolist()]
                        pareto_line_colors = ["black"]
                        pareto_line_widths = [2.0]
                        pareto_points_x = pareto_x.tolist()
                        pareto_points_y = pareto_y.tolist()
                        pareto_points_c = pareto_c.tolist()
                        pareto_point_line_colors = ["black"] * int(pareto_x.size)
                        pareto_point_line_widths = [0.7] * int(pareto_x.size)
                    if best_index is not None:
                        best_plot_pos_arr = np.flatnonzero(plot_indices == best_index)
                        if best_plot_pos_arr.size > 0:
                            plot_pos = int(best_plot_pos_arr[0])
                            best_rows = [BestRowDescriptor(row_index=best_index, color_value=float(store.arrays[color_col][best_index]))]
                            best_x = x_plot[plot_pos : plot_pos + 1].tolist()
                            best_y = y_plot[plot_pos : plot_pos + 1].tolist()
                            best_c = transform_values_inplace(
                                store.arrays[color_col][np.array([best_index], dtype=np.int64)],
                                color_log,
                            ).tolist()
                            best_line_colors = ["red"]
            else:
                candidate_row_indices = plot_indices
                selection_count = int(candidate_row_indices.size)
                best_index, tied_indices = select_best_candidates(store.arrays, candidate_row_indices, sort_specs)
                tied_count = int(max(tied_indices.size - 1, 0))
                if best_index is not None:
                    best_plot_pos_arr = np.flatnonzero(plot_indices == best_index)
                    if best_plot_pos_arr.size > 0:
                        plot_pos = int(best_plot_pos_arr[0])
                        best_rows = [BestRowDescriptor(row_index=best_index, color_value=float(store.arrays[color_col][best_index]))]
                        best_x = x_plot[plot_pos : plot_pos + 1].tolist()
                        best_y = y_plot[plot_pos : plot_pos + 1].tolist()
                        best_c = transform_values_inplace(
                            store.arrays[color_col][np.array([best_index], dtype=np.int64)],
                            color_log,
                        ).tolist()
                        best_line_colors = ["red"]

            pareto_line_source.data = dict(xs=pareto_line_xs, ys=pareto_line_ys, line_color=pareto_line_colors, line_width=pareto_line_widths)
            pareto_points_source.data = dict(x=pareto_points_x, y=pareto_points_y, c=pareto_points_c, line_color=pareto_point_line_colors, line_width=pareto_point_line_widths)
            if show_best_match_rows:
                best_source.data = dict(x=best_x, y=best_y, c=best_c, line_color=best_line_colors)
            else:
                best_source.data = dict(x=[], y=[], c=[], line_color=[])
            best_rows = sort_best_rows_by_z_order(best_rows, zorder_select.value)
            update_best_row_display(best_rows, filter_match_count, selection_count, sort_specs, tied_count, grouped_pareto_column)

            notes: list[str] = []
            if baseline_hidden:
                notes.append("baseline square hidden by current x/y log or non-finite values")
            filtered_hidden = max(filter_match_count - int(plot_indices.size), 0)
            if filtered_hidden:
                notes.append(f"{filtered_hidden:,} filtered row(s) hidden by current x/y/color log or non-finite values")
            if pareto_enabled and user_pareto_requested and USER_PARETO_COLUMN not in store.arrays:
                notes.append(f"user-based Pareto mode skipped because the dataset has no '{USER_PARETO_COLUMN}' column")
            elif pareto_enabled and user_pareto_requested and user_hidden_count:
                notes.append(f"{user_hidden_count:,} plotted row(s) skipped in user-based Pareto mode because '{USER_PARETO_COLUMN}' is non-finite")
            if pareto_enabled and discrete_pareto_requested and (not use_user_paretos) and unique_color_count > 7:
                notes.append(f"discretized-color Pareto mode skipped because the color axis has {unique_color_count:,} unique values (> 7)")
            elif use_user_paretos:
                notes.append(f"drawing {len(user_pareto_results):,} Pareto frontier(s), one for each unique {USER_PARETO_COLUMN} value")
                notes.append("rows outside the current filters are hidden in user-based Pareto mode")
            elif use_discrete_paretos:
                notes.append(f"drawing {len(discrete_pareto_results):,} Pareto frontier(s), one for each unique color value")
            if not show_best_match_rows and best_rows:
                notes.append("best-row star markers hidden by toggle")
            elapsed = time.perf_counter() - t0
            note_suffix = " " + "; ".join(notes) + "." if notes else ""
            status.text = (
                f"Rendered <b>{render['n_rows']:,}</b> filtered plottable rows over a <b>{background_entry.n_rows:,}</b>-row "
                f"full-dataset background into {plot_width}x{plot_height} pixels in <b>{elapsed:.3f}s</b>. "
                #f"Filtered non-empty pixels: <b>{render['n_nonzero_pixels']:,}</b>. "
                #f"Full-background non-empty pixels: <b>{background_entry.n_nonzero_pixels:,}</b>. "
                #f"Max filtered points in one base pixel: <b>{render['max_count_per_pixel']:,}</b>. "
                f"Color scale uses <b>{full_color_count:,}</b> full-dataset valid values. "
                f"{('Per-user Pareto frontier points' if use_user_paretos else ('Per-color Pareto frontier points' if use_discrete_paretos else ('Pareto frontier points' if pareto_enabled else 'Filtered plottable points')))}: <b>{selection_count:,}</b>. "
                f"Best-row marker(s): <b>{len(best_rows):,}</b>.{note_suffix}"
                f"<p><b>The <span style=\'color:red;\'>red square</span> is the baseline design.</b></p>"
                
            )

        update_button.on_click(do_update)
        reset_plot_button.on_click(reset_plot_view)
        subgroup_visibility_checks.on_change("active", lambda attr, old, new: refresh_best_row_visibility_only())

        left_controls = column(
            row(x_select, x_pareto_select, y_select, y_pareto_select, color_select, zorder_select, column(update_button, load_filter_json_button)),
            row(pareto_checkbox, discrete_pareto_checkbox, user_pareto_checkbox, show_best_match_checkbox),
            row(log_checks),
            filters_header,
            row(add_filter_button),
            filters_box,
            sort_header,
            row(*sort_row_widgets),
        )
        json_panel = column(filter_json, filter_json_help, filter_json_status, width=JSON_PANEL_WIDTH)
        controls = row(left_controls, json_panel, align="start")

        right_panel_children = [best_row_summary]
        if toggleable_groups:
            right_panel_children.extend([subgroup_visibility_header, subgroup_visibility_checks])
        right_panel_children.extend([subgroup_visibility_note, best_row_table])
        right_panel = column(*right_panel_children, sizing_mode="fixed")

        plot_controls = row(reset_plot_button, download_scale_select, download_png_button)
        plot_column = column(fig, plot_controls)
        doc.add_root(column(title, controls, status, row(plot_column, right_panel), sizing_mode="stretch_width"))
        doc.title = "DreamRAM Explorer"
        refresh_filter_widgets()
        sync_json_from_ui(filters=[])
        doc.add_next_tick_callback(do_update)

    return make_document

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    csv_path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.start_advanced:
        print("Ignoring deprecated --start-advanced/--advanced flag because the Advanced toggle has been removed.", file=sys.stderr)
    if args.simple_axes_csv is not None:
        print("Ignoring deprecated --simple-axes-csv option; use --subgroups-json instead.", file=sys.stderr)

    print("Loading CSV...")
    store = load_float_csv(csv_path, dtype=args.dtype)
    print("Assigning Labels...")
    labels = LabelManager.build(store.columns, args.label_csv)
    group_manager = ColumnGroupManager.build(store.columns, args.subgroups_json)
    print("Identifying Baseline...")
    baseline_info = find_baseline_info(store)
    print("Making App...")
    app = make_document_factory(
        store,
        labels,
        group_manager=group_manager,
        baseline_info=baseline_info,
        plot_width=args.plot_width,
        plot_height=args.plot_height,
        palette_name=args.colormap,
    )

    server = Server({"/": app}, port=args.port, allow_websocket_origin=[f"localhost:{args.port}"])
    server.start()

    url = f"http://localhost:{args.port}/"
    print(f"Bokeh app running at {url}")
    if not args.no_browser:
        server.io_loop.add_callback(lambda: webbrowser.open(url))
    server.io_loop.start()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
