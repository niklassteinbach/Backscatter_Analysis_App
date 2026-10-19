#!/usr/bin/env python3
"""
Backscatter Analysis App

Desktop application for analyzing backscatter time-series data with two detrending methods:
- Polynomial detrending followed by FFT and cosine analysis
- Logistic detrending followed by signal and FFT analysis

The analysis functions calculate the growth trend, isolate the oscillating signal,
and create the numerical results and plots. The graphical interface handles data
loading, analysis settings, grouping, plotting, previews, and data export.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib import colors as mpl_colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import circmean, circstd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser


APP_VERSION = "0.3.12"


# ============================================================================
# Core analysis calculations
# ============================================================================

def polynomial_regression(y: np.ndarray, kernel_size: int, poly: int = 4) -> tuple:
    """
    Perform polynomial regression on the input signal and return the polynomial regression values,
    oscillation signal and smoothed signal.
    Parameters:
    - y: Input signal (1D numpy array)
    - kernel_size: Size of the smoothing kernel (int)
    - poly: Degree of the polynomial for regression (int, default=4)
    Returns:
    - smoothed_signal: Smoothed oscillation signal (1D numpy array)
    - signal: Oscillation signal (1D numpy array)
    - reg_predict: Polynomial regression values (1D numpy array)
    """

    kernel = np.ones(kernel_size) / kernel_size  # Smoothing kernel

    y_reg = np.array(np.arange(len(y))).reshape(-1, 1)  # Time variable for regression
    pf = PolynomialFeatures(poly)  # Polynomial feature transformer
    y_reg = pf.fit_transform(y_reg)  # Transform time variable to polynomial features
    reg_fit = LinearRegression().fit(y_reg, y)  # Fit polynomial regression
    reg_predict = reg_fit.predict(y_reg)  # Predict polynomial regression values
    signal = y - reg_predict  # residual signal
    signal -= signal.mean()  # Center the signal around zero
    smoothed_signal = np.convolve(signal, kernel, mode="same")  # Smooth the signal
    smoothed_signal[-int(kernel_size / 2):] = np.nan  # Remove edge effects
    smoothed_signal[:int(kernel_size / 2)] = np.nan  # Remove edge effects
    return smoothed_signal, signal, reg_predict


def phase_to_radians(phases_hours, period_hours=24.0):
    return (phases_hours / period_hours) * 2 * np.pi


def radians_to_hours(radians, period_hours=24.0):
    return (radians / (2 * np.pi)) * period_hours


def cosine_peak_phase_hours(phase_shift_hours, period_hours):
    """Return the cosine maximum within one fitted period, measured from t=0.

    The fitted model is ``A*cos(2*pi*f*t + phase)``. A maximum occurs when
    ``2*pi*f*t + phase = 2*pi*k``. Because ``phase_shift_hours`` is
    ``phase/(2*pi) * period``, the peak time within one cycle is simply
    ``(-phase_shift_hours) modulo period``. The result is therefore in the
    interval [0, period).
    """
    phase_shift_hours = float(phase_shift_hours)
    period_hours = float(period_hours)
    if not np.isfinite(phase_shift_hours) or not np.isfinite(period_hours) or period_hours <= 0:
        return np.nan
    peak_phase = (-phase_shift_hours) % period_hours
    # Avoid reporting a value numerically indistinguishable from one full period.
    if np.isclose(peak_phase, period_hours, rtol=0.0, atol=max(1e-12, abs(period_hours) * 1e-12)):
        peak_phase = 0.0
    return float(peak_phase)


def circular_mean_hours(phases_hours, period_hours=24.0):
    radians = phase_to_radians(phases_hours, period_hours)
    mean_rad = circmean(radians)
    if mean_rad < 0:
        mean_rad += 2 * np.pi
    return radians_to_hours(mean_rad, period_hours)


def circular_std_hours(phases_hours, period_hours=24.0):
    radians = phase_to_radians(phases_hours, period_hours)
    std_rad = circstd(radians)
    return radians_to_hours(std_rad, period_hours)


def circular_sem_hours(phases_hours, period_hours=24.0):
    std = circular_std_hours(phases_hours, period_hours)
    n = len(phases_hours)
    return std / np.sqrt(n)


def fourier_transform(mean_smooth: np.ndarray, kernel_size: int, zero_padding: int) -> tuple:
    """Calculate the FFT using a fixed 20-second measurement interval."""
    signal = np.concatenate(
        (
            mean_smooth[~np.isnan(mean_smooth)],
            np.zeros(len(mean_smooth[~np.isnan(mean_smooth)]) * zero_padding),
        )
    )
    T = 20 / 3600
    N = signal.shape[0]
    N_orig = mean_smooth[~np.isnan(mean_smooth)].shape[0]
    yf = rfft(signal)
    xf = rfftfreq(N, T)
    dominant_idx = np.argmax(2 / N_orig * np.abs(yf))
    dominant_freq = xf[dominant_idx]
    dominant_period = 1 / dominant_freq
    dominant_amplitude = 2 / N_orig * np.abs(yf[dominant_idx])
    phase_rad = np.angle(yf[dominant_idx])
    phase_shift_hours = radians_to_hours(phase_rad, period_hours=dominant_period)
    phase_correction = kernel_size * 20 / 60 / 60 / 2
    phase_shift_hours = phase_shift_hours + phase_correction
    corrected_phase_rad = phase_to_radians(phase_shift_hours, period_hours=dominant_period)
    corrected_amplitude = 2 / N_orig * np.abs(yf)[:len(mean_smooth)]
    xf_corrected = xf[:len(mean_smooth)]
    return (
        dominant_amplitude,
        dominant_freq,
        dominant_period,
        phase_shift_hours,
        corrected_phase_rad,
        corrected_amplitude,
        xf_corrected,
    )


def cosine_fit(
    time: np.ndarray,
    mean_smooth: np.ndarray,
    name: str,
    dominant_amplitude,
    dominant_freq,
    corrected_phase_rad,
):
    """Fit a cosine curve to the smoothed oscillation signal."""
    approximation = dominant_amplitude * np.cos(
        2 * np.pi * dominant_freq * time + corrected_phase_rad
    )

    def cosine_wave(x, A, f, phase):
        return A * np.cos(2 * np.pi * f * x + phase)

    initial_guess = [dominant_amplitude, dominant_freq, corrected_phase_rad]
    param_bounds = ([0, 0, -2 * np.pi], [np.inf, np.inf, 2 * np.pi])

    try:
        params, covariance = curve_fit(
            cosine_wave,
            time[~np.isnan(mean_smooth)],
            mean_smooth[~np.isnan(mean_smooth)],
            p0=initial_guess,
            bounds=param_bounds,
            maxfev=10000,
        )

        A_fit, f_fit, phase_fit = params
        wavelength = 1 / f_fit if f_fit != 0 else np.inf
        y_fit = cosine_wave(time, A_fit, f_fit, phase_fit)
        time_shift = radians_to_hours(phase_fit, period_hours=wavelength)

        if time_shift > wavelength / 2:
            time_shift = time_shift - wavelength
        elif time_shift < -wavelength / 2:
            time_shift = time_shift + wavelength

        time_shift_rad = phase_to_radians(time_shift, period_hours=wavelength)
        return A_fit, wavelength, time_shift, time_shift_rad, y_fit, covariance

    except RuntimeError:
        return approximation


def normalize_logistic_residual(y: np.ndarray, reg_predict: np.ndarray, mode: str = "mean_centered") -> np.ndarray:
    """Convert the logistic-fit residual into the selected signal representation.

    ``mean_centered`` keeps the original behavior: subtract the logistic fit from
    the measured biomass and center the residual around zero by subtracting its
    arithmetic mean.

    ``fitted_biomass`` divides the residual by the fitted biomass, producing a
    dimensionless relative signal. Values for which the fitted biomass is
    effectively zero are left undefined to avoid unstable division.
    """
    y = np.asarray(y, dtype=float)
    reg_predict = np.asarray(reg_predict, dtype=float)
    residual = y - reg_predict

    if mode == "fitted_biomass":
        signal = np.full_like(residual, np.nan, dtype=float)
        finite_fit = reg_predict[np.isfinite(reg_predict)]
        if finite_fit.size == 0:
            return signal
        scale = float(np.nanmax(np.abs(finite_fit)))
        threshold = max(np.finfo(float).eps, scale * 1e-6)
        valid = np.isfinite(residual) & np.isfinite(reg_predict) & (np.abs(reg_predict) > threshold)
        signal[valid] = residual[valid] / reg_predict[valid]
        return signal

    mean_val = np.nanmean(residual)
    if not np.isfinite(mean_val):
        mean_val = 0.0
    return residual - mean_val


def signal_normalization_label(mode: str) -> str:
    """Short axis/export label for the selected signal normalization."""
    return "Relative signal [fraction of fitted biomass]" if mode == "fitted_biomass" else "Signal [a.u.]"


def signal_column_names(mode: str):
    """Return clear raw/smoothed column names for exports."""
    if mode == "fitted_biomass":
        return "Relative signal [fraction of fitted biomass]", "Smoothed relative signal [fraction of fitted biomass]"
    return "Signal [a.u.]", "Smoothed signal [a.u.]"


def logistic_regression_adaptive_full(
    y,
    kernel,
    poly=4,
    smooth_sigma=2,
    slope_fraction=0.05,
    normalization_mode="mean_centered",
):
    """
    Fit a logistic growth curve with adaptive lag-phase detection.
    The residual is used as the oscillation signal.
    """
    y = np.array(y)
    x = np.arange(len(y))

    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)
    dy = np.diff(y_smooth, prepend=y_smooth[0])

    threshold = slope_fraction * np.nanmax(dy)
    lag_candidates = np.where(dy > threshold)[0]
    if len(lag_candidates) == 0:
        lag_estimate = len(y) // 4
    else:
        lag_estimate = lag_candidates[0]

    def logistic(x, K, r, x0, v):
        return K / (1 + np.exp(-r * (x - x0))) ** (1 / v)

    try:
        p0 = [np.nanmax(y), 0.01, lag_estimate, 1]
        bounds = ([0, 0, 0, 0.1], [np.inf, 1, len(y), 10])
        params, _ = curve_fit(
            logistic,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        reg_predict = logistic(x, *params)
    except Exception:
        params = None
        reg_predict = np.full_like(y, np.nan)

    signal = normalize_logistic_residual(y, reg_predict, normalization_mode)
    smoothed_signal = np.convolve(signal, kernel, mode="same")
    return smoothed_signal, signal, reg_predict, params


def logistic_regression(y, kernel, poly=4, normalization_mode="mean_centered"):
    """Fit a logistic growth curve and return fit, residual signal, and smoothed signal."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5 or np.all(np.isnan(y)):
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    x = np.arange(n)
    y_safe = y.copy()
    if np.any(np.isnan(y_safe)):
        nan_mask = np.isnan(y_safe)
        y_safe[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            y_safe[~nan_mask],
        )

    def logistic(x, K, r, x0, v):
        return K / (1 + np.exp(-r * (x - x0))) ** (1 / v)

    try:
        p0 = [np.nanmax(y_safe), 0.01, n / 2, 1]
        bounds = ([0, 0, 0, 0.1], [np.inf, 1, n, 10])
        params, _ = curve_fit(
            logistic,
            x,
            y_safe,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        reg_predict = logistic(x, *params)
    except Exception:
        reg_predict = np.full(n, np.nan)

    signal = normalize_logistic_residual(y, reg_predict, normalization_mode)

    if len(kernel) > 0 and len(signal) > 0:
        smoothed_signal = np.convolve(signal, kernel, mode="same")
    else:
        smoothed_signal = np.full(n, np.nan)

    return smoothed_signal, signal, reg_predict


# ============================================================================
# Data model and helper functions
# ============================================================================

@dataclass
class SampleConfig:
    sample_id: str
    group: str
    plot_label: str
    color: str
    include_in_plots: bool = True


@dataclass
class AnalysisSettings:
    method: str = "polynomial"
    analysis_start_h: float = 0.0
    analysis_end_h: float = 216.0
    polynomial_degree: int = 4
    logistic_start_min_h: float = 0.0
    logistic_start_max_h: float = 0.0
    logistic_end_min_h: float = 216.0
    logistic_end_max_h: float = 216.0
    logistic_signal_start_h: float = 0.0
    logistic_signal_end_h: float = 216.0
    logistic_grid_points: int = 7
    logistic_signal_normalization: str = "mean_centered"
    logistic_visual_fit_enabled: bool = False
    logistic_visual_fit_start_h: float = 0.0
    logistic_visual_fit_end_h: float = 48.0
    kernel_size: int = 1200
    zero_padding: int = 4
    measurement_interval_s: float = 20.0
    auto_measurement_interval: bool = True
    smooth_sigma: float = 2.0
    slope_fraction: float = 0.05


@dataclass
class PlotSettings:
    output_format: str = "both"
    figure_width: float = 9.0
    figure_height: float = 4.8
    dpi: int = 600
    line_width: float = 2.0
    line_style: str = "-"
    fit_line_width: float = 2.0
    fit_line_style: str = "--"
    # Transparency defaults keep the biomass visible while making processed signals stand out.
    # Raw residuals are shown more faintly than the smoothed signal and fitted curves.
    biomass_alpha: float = 0.65
    raw_alpha: float = 0.25
    fit_alpha: float = 0.95
    smooth_alpha: float = 0.90
    cosine_alpha: float = 0.95
    # Contrasting default fit colors make the regression/cosine curves visible on top
    # of the sample/group color. Both are editable in the GUI.
    fit_color: str = "#111111"
    cosine_color: str = "#111111"
    font_size: float = 11.0
    title_font_size: float = 12.0
    legend_font_size: float = 9.0
    x_start_h: Optional[float] = None
    x_end_h: Optional[float] = None
    biomass_y_min: Optional[float] = None
    biomass_y_max: Optional[float] = None
    signal_y_min: Optional[float] = None
    signal_y_max: Optional[float] = None
    fft_x_min: float = 0.0
    fft_x_max: float = 0.3
    fft_y_min: Optional[float] = None
    fft_y_max: Optional[float] = None
    show_12h_24h: bool = True
    major_tick_h: float = 24.0
    minor_tick_h: float = 12.0
    show_grid: bool = True
    grid_alpha: float = 0.30
    minor_grid_alpha: float = 0.14
    entrainment_end_h: float = 0.0
    light_hours: float = 12.0
    dark_hours: float = 12.0
    show_raw_signal: bool = True
    show_smoothed_signal: bool = True
    show_cosine: bool = True
    # Optional display-only crop for the cosine line. These limits never change
    # the cosine fit, FFT, period, amplitude, or phase calculations.
    cosine_plot_start_h: Optional[float] = None
    cosine_plot_end_h: Optional[float] = None
    # Optional visual boundary between the display-only logistic signal and the
    # real analytical signal. This is a plotting aid only.
    show_visual_only_divider: bool = True
    show_extrema: bool = True
    annotate_extrema: bool = True
    peak_width: int = 100
    peak_distance: int = 1000
    annotate_fft_period: bool = True
    legend_below: bool = False
    legend_ncol: int = 2
    separate_legend: bool = True
    generate_individual: bool = True
    generate_overview_grids: bool = True
    generate_replicate_overlays: bool = True
    generate_condition_means: bool = True
    generate_all_conditions: bool = True
    generate_by_replicate: bool = True
    overview_ncols: int = 3
    generate_fft_period_view: bool = True
    # Empty tuple means no group restriction. In the GUI the loaded groups are
    # selected by default; this field stores an explicit multi-group selection.
    selected_groups: Optional[Tuple[str, ...]] = None
    # Optional custom group combinations. Each tuple produces a separate set of
    # comparison plots, e.g. (("WT", "Treatment"), ("WT", "Mutant")).
    custom_group_comparisons: Tuple[Tuple[str, ...], ...] = ()


def safe_float(value: str, default: Optional[float] = None) -> Optional[float]:
    value = str(value).strip().replace(",", ".")
    if value == "":
        return default
    return float(value)


def safe_int(value: str, default: int) -> int:
    value = str(value).strip()
    return default if value == "" else int(value)


def r2_score_finite(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("-inf")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("-inf")
    return float(1 - ss_res / ss_tot)


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return text.strip("._-") or "sample"


def infer_group(sample_name: str) -> str:
    """Best-effort replicate grouping. The GUI always allows manual correction."""
    patterns = [
        r"^(.*?)-\d+_\d+$",              # sample-1_2
        r"^(.*?)_\d+_\d+$",              # sample_1_2
        r"^(.*?)[_-]rep(?:licate)?[_-]?\d+$",
        r"^(.*?)[.-]\d+$",                # sample.1 / sample-1
    ]
    for pattern in patterns:
        match = re.match(pattern, sample_name, flags=re.IGNORECASE)
        if match and match.group(1):
            return match.group(1)
    return sample_name


def default_sample_configs(names: List[str]) -> List[SampleConfig]:
    groups = []
    for name in names:
        group = infer_group(name)
        if group not in groups:
            groups.append(group)

    cmap = plt.get_cmap("tab20")
    group_colors = {
        group: mpl_colors.to_hex(cmap(i % cmap.N))
        for i, group in enumerate(groups)
    }
    return [
        SampleConfig(
            sample_id=name,
            group=infer_group(name),
            plot_label=name,
            color=group_colors[infer_group(name)],
        )
        for name in names
    ]


def filter_plot_configs(configs: List[SampleConfig], ps: PlotSettings) -> List[SampleConfig]:
    """Return samples that should contribute to plots and plotted group statistics.

    All samples are still analyzed and exported. This filter affects only plots,
    plotted means/medians, plotted replicate overlays, and plot legends.
    """
    # None is the backward-compatible programmatic default = all groups.
    # An explicit empty tuple from the GUI means the user selected no groups.
    if ps.selected_groups is None:
        return [c for c in configs if c.include_in_plots]
    selected = set(ps.selected_groups)
    return [c for c in configs if c.include_in_plots and c.group in selected]


def combined_time_bounds(combined_data_file: str) -> Tuple[float, float]:
    """Return finite min/max Relative Time across all samples in a combined CSV."""
    df = pd.read_csv(combined_data_file)
    cols = [c for c in df.columns if str(c).endswith("_Time_Relative")]
    values = []
    for col in cols:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return 0.0, 0.0
    return float(min(v.min() for v in values)), float(max(v.max() for v in values))


def resolve_experiment_folder(selected_folder: str) -> Tuple[Path, Path, str]:
    selected = Path(selected_folder).expanduser().resolve()
    if not selected.is_dir():
        raise FileNotFoundError(f"Folder not found: {selected}")

    if selected.name.lower() == "excel exports":
        experiment_dir = selected.parent
        excel_dir = selected
    else:
        experiment_dir = selected
        excel_dir = experiment_dir / "excel exports"
        if not excel_dir.is_dir():
            # Compatibility with older experiments that kept xlsx files in root.
            if any(experiment_dir.glob("*.xlsx")):
                excel_dir = experiment_dir
            else:
                raise FileNotFoundError(
                    f"Could not find an 'excel exports' folder inside:\n{experiment_dir}"
                )

    return experiment_dir, excel_dir, experiment_dir.name


def _read_biolector_backscatter_workbook(path: Path, log=print):
    """Read a multi-well BioLector export and return one time/biomass pair per described well.

    The HHU publication files store the well-to-sample mapping in ``LayoutData``
    and the biomass channel in ``Channel1``. ``Channel1`` contains both calibrated
    and raw blocks. This importer deliberately selects columns marked ``Raw`` so
    the values match the supplementary readme, which states that only raw values
    were used for the publication analysis.

    Source time is stored in hours and is kept in hours in the combined CSV used
    internally by this app. No shifting, trimming, smoothing, normalization, or
    other analysis is performed during import.
    """
    with pd.ExcelFile(path) as excel:
        sheets = set(excel.sheet_names)
        if not {"LayoutData", "Channel1"}.issubset(sheets):
            return [], pd.DataFrame()

        layout = pd.read_excel(excel, sheet_name="LayoutData")
        channel = pd.read_excel(excel, sheet_name="Channel1", header=None)

    required_layout = {"Well", "Description"}
    if not required_layout.issubset(layout.columns):
        raise ValueError(
            f"BioLector workbook {path.name} has LayoutData but is missing "
            f"columns {sorted(required_layout)}."
        )
    if channel.shape[0] < 5 or channel.shape[1] < 3:
        raise ValueError(f"BioLector Channel1 sheet in {path.name} is incomplete.")

    # Channel1 structure in these BioLector exports:
    # row 0 = column names, row 1 = channel name, row 2 = Calibrated/Raw,
    # row 3 = well ID, rows 4+ = measurements. Column 1 stores Time[h].
    raw_col_by_well = {}
    for col_idx in range(2, channel.shape[1]):
        data_kind = str(channel.iat[2, col_idx]).strip().lower()
        well = channel.iat[3, col_idx]
        if data_kind == "raw" and pd.notna(well):
            raw_col_by_well[str(well).strip()] = col_idx

    if not raw_col_by_well:
        raise ValueError(f"No raw Channel1 well columns were found in {path.name}.")

    time_h = pd.to_numeric(channel.iloc[4:, 1], errors="coerce").reset_index(drop=True)
    result = pd.DataFrame()
    names = []

    for _, row in layout.iterrows():
        well = row.get("Well")
        description = row.get("Description")
        if pd.isna(well) or pd.isna(description):
            continue
        well = str(well).strip()
        name = str(description).strip()
        if not name or well not in raw_col_by_well:
            continue

        biomass = pd.to_numeric(
            channel.iloc[4:, raw_col_by_well[well]], errors="coerce"
        ).reset_index(drop=True)
        pair = pd.DataFrame({
            f"{name}_Time_Relative": time_h,
            f"{name}_Biomass": biomass,
        })
        valid = np.isfinite(pair[f"{name}_Time_Relative"].to_numpy(float)) & np.isfinite(
            pair[f"{name}_Biomass"].to_numpy(float)
        )
        if not valid.any():
            log(f"Warning: BioLector well {well} ({name}) contains no usable raw data; skipped.")
            continue

        result = pd.concat([result, pair], axis=1)
        names.append(name)

    log(
        f"BioLector import: {path.name} -> {len(names)} described wells from raw Channel1 "
        f"(730-nm biomass channel in the HHU supplementary files)."
    )
    log(
        "BioLector import only reformats the source data. The full time series and original "
        "time axis are retained; no detrending, smoothing, normalization, or 84-h cutoff is applied."
    )
    return names, result


def combine_data_from_experiment(selected_folder: str, log=print):
    """
    Read all Excel files belonging to one experiment and combine their backscatter data.

    Two input layouts are supported:
    1. The normal one-sample-per-Excel export used by this app.
    2. Multi-well BioLector exports with ``LayoutData`` and ``Channel1`` sheets,
       including the raw HHU Köbler et al. 2024 supplementary backscatter files.

    The combined CSV is written beside the source Excel files. Import only changes
    layout; the numerical backscatter values are not normalized or otherwise analyzed.
    """
    experiment_dir, excel_dir, experiment_name = resolve_experiment_folder(selected_folder)

    file_list = sorted([p for p in excel_dir.iterdir() if p.suffix.lower() == ".xlsx"])
    if not file_list:
        raise FileNotFoundError(f"No .xlsx files found in: {excel_dir}")

    output_file = excel_dir / f"{experiment_name}_combined_data_log.csv"
    data = pd.DataFrame()
    kept_names: List[str] = []

    for path in file_list:
        log(f"Reading {path.name}")
        with pd.ExcelFile(path) as excel:
            sheet_names = excel.sheet_names

        # BioLector multi-well exports are recognized by their structural sheets.
        if "LayoutData" in sheet_names and "Channel1" in sheet_names:
            bio_names, bio_data = _read_biolector_backscatter_workbook(path, log=log)
            if not bio_data.empty:
                # Avoid silently overwriting duplicate sample IDs if a folder contains
                # more than one multi-well workbook with the same descriptions.
                duplicates = set(kept_names).intersection(bio_names)
                if duplicates:
                    raise ValueError(
                        "Duplicate sample descriptions across BioLector workbooks: "
                        + ", ".join(sorted(duplicates))
                        + ". Load each BioLector experiment folder separately."
                    )
                data = pd.concat([data, bio_data.reset_index(drop=True)], axis=1)
                kept_names.extend(bio_names)
            continue

        # Standard one-sample-per-file import used for the user's experimental exports.
        matching_sheets = [
            sheet for sheet in sheet_names
            if re.search("backscatter", sheet, flags=re.IGNORECASE)
        ]
        if not matching_sheets:
            log(f"Warning: no backscatter sheet found in {path.name}; skipped.")
            continue

        name = re.split(rf"{re.escape(experiment_name)}_", path.stem)[-1]
        with pd.ExcelFile(path) as excel:
            sub_data = pd.read_excel(
                excel,
                sheet_name=matching_sheets[0],
                skiprows=1,
            ).iloc[:, 1:]

        sub_data = sub_data.rename(
            columns={
                "Relative Time [s]": f"{name}_Time_Relative",
                "Raw backscatter [a.u.]": f"{name}_Biomass",
                "Raw backscatter [biomass a.u.]": f"{name}_Biomass",
            }
        )

        time_column = f"{name}_Time_Relative"
        biomass_column = f"{name}_Biomass"
        if time_column not in sub_data.columns or biomass_column not in sub_data.columns:
            log(
                f"Warning: required columns not found in {path.name}; skipped. "
                f"Columns: {sub_data.columns.tolist()}"
            )
            continue

        sub_data[time_column] = pd.to_numeric(
            sub_data[time_column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ) / 3600
        sub_data[biomass_column] = pd.to_numeric(
            sub_data[biomass_column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

        data = pd.concat(
            [data, sub_data[[time_column, biomass_column]].reset_index(drop=True)],
            axis=1,
        )
        kept_names.append(name)

    if data.empty:
        raise ValueError("No usable backscatter data could be combined.")

    data.to_csv(output_file, index=False)
    log(f"Combined data saved to: {output_file}")
    return kept_names, str(output_file), str(experiment_dir), experiment_name


def load_combined_sample(combined_df: pd.DataFrame, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
    time_col = f"{sample_id}_Time_Relative"
    biomass_col = f"{sample_id}_Biomass"
    if time_col not in combined_df.columns or biomass_col not in combined_df.columns:
        raise KeyError(f"Missing columns for {sample_id}")

    df = combined_df[[time_col, biomass_col]].copy()
    for col in [time_col, biomass_col]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
    df = df.dropna(subset=[time_col, biomass_col]).sort_values(time_col)
    return df[time_col].to_numpy(float), df[biomass_col].to_numpy(float)


def candidate_values(min_h: float, max_h: float, grid_points: int) -> List[float]:
    if max_h < min_h:
        min_h, max_h = max_h, min_h
    if math.isclose(min_h, max_h):
        return [float(min_h)]
    return list(np.linspace(min_h, max_h, max(2, int(grid_points))))


def polynomial_fixed_degree(
    time: np.ndarray,
    biomass: np.ndarray,
    start_h: float,
    end_h: float,
    kernel_size: int,
    degree: int = 4,
):
    """
    Apply polynomial detrending once using the selected polynomial degree.

    The selected time window includes both the lower and upper boundaries.
    """
    degree = int(degree)
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")

    mask = (time >= start_h) & (time <= end_h)
    idx = np.flatnonzero(mask)
    if len(idx) <= kernel_size:
        raise ValueError(
            f"Polynomial window has {len(idx)} points, but kernel_size={kernel_size}."
        )

    t = time[idx]
    y = biomass[idx]

    # Apply polynomial detrending once with the manually selected degree.
    smooth, signal, fit = polynomial_regression(
        y,
        kernel_size=kernel_size,
        poly=degree,
    )
    r2 = r2_score_finite(y, fit)  # reporting only; does not affect the fit

    full_fit = np.full(len(time), np.nan)
    full_signal = np.full(len(time), np.nan)
    full_smooth = np.full(len(time), np.nan)
    full_fit[idx] = fit
    full_signal[idx] = signal
    full_smooth[idx] = smooth

    return {
        "fit": full_fit,
        "signal": full_signal,
        "smooth": full_smooth,
        "r2": r2,
        "degree": degree,
        "fit_start_h": float(t[0]),
        "fit_end_h": float(t[-1]),
        "degree_scores": pd.DataFrame([{"degree": degree, "r2": r2}]),
    }


def logistic_best_window(
    time: np.ndarray,
    biomass: np.ndarray,
    start_min_h: float,
    start_max_h: float,
    end_min_h: float,
    end_max_h: float,
    kernel_size: int,
    grid_points: int,
    smooth_sigma: float,
    slope_fraction: float,
    normalization_mode: str = "mean_centered",
    signal_start_h: float = None,
    signal_end_h: float = None,
):
    kernel = np.ones(kernel_size) / kernel_size
    starts = candidate_values(start_min_h, start_max_h, grid_points)
    ends = candidate_values(end_min_h, end_max_h, grid_points)

    best = None
    rows = []
    for start_h in starts:
        for end_h in ends:
            if end_h <= start_h:
                continue

            fit_indices = np.flatnonzero((time >= start_h) & (time <= end_h))
            if len(fit_indices) == 0:
                continue
            l_lim = int(fit_indices[0])
            u_lim = int(fit_indices[-1]) + 1  # Python slice end is exclusive

            # Use only the selected interval to estimate the logistic parameters.
            if u_lim <= l_lim:
                continue
            y_window = biomass[l_lim:u_lim]
            if len(y_window) <= max(5, 2 * kernel_size + 2):
                continue

            smooth, signal, fit, params = logistic_regression_adaptive_full(
                y_window,
                kernel,
                poly=4,
                smooth_sigma=smooth_sigma,
                slope_fraction=slope_fraction,
                normalization_mode=normalization_mode,
            )
            score = r2_score_finite(y_window, fit)
            rows.append(
                {
                    "requested_start_h": start_h,
                    "requested_end_h": end_h,
                    "actual_start_h": float(time[l_lim]),
                    "actual_end_h": float(time[u_lim - 1]),
                    "n_points": len(y_window),
                    "r2": score,
                }
            )
            if not np.isfinite(score):
                continue
            if best is None or score > best[0]:
                best = (score, start_h, end_h, l_lim, u_lim, smooth, signal, fit, params)

    if best is None:
        raise ValueError(
            "No valid logistic fit was found. Try a wider fit window, a smaller kernel, "
            "or fewer restrictions on the start/end search."
        )

    r2, requested_start, requested_end, l_lim, u_lim, _fit_smooth, _fit_signal, fit, params = best
    if params is None:
        raise ValueError("The selected logistic fit did not return valid parameters.")

    # The fit window and the analytical-signal window are intentionally separate.
    # The fit window above is used only to estimate K, r, x0 and v.  The selected
    # logistic curve is then evaluated across the requested signal-analysis range,
    # so FFT/cosine analysis is not artificially shortened to the fitting interval.
    if signal_start_h is None:
        signal_start_h = float(time[l_lim])
    if signal_end_h is None:
        signal_end_h = float(time[-1])
    if signal_end_h < signal_start_h:
        signal_start_h, signal_end_h = signal_end_h, signal_start_h

    analysis_idx = np.flatnonzero((time >= signal_start_h) & (time <= signal_end_h))
    if len(analysis_idx) == 0:
        raise ValueError("The logistic signal-analysis window contains no data points.")
    if len(analysis_idx) <= kernel_size:
        raise ValueError(
            f"Logistic signal-analysis window has {len(analysis_idx)} points, but "
            f"kernel_size={kernel_size}. Widen the signal window or reduce the smoothing kernel."
        )

    # The fitted window used x=0 at l_lim.  Preserve that coordinate system when
    # evaluating the same curve before/after the fit interval.
    x_analysis = analysis_idx.astype(float) - float(l_lim)
    K, r, x0, v = [float(x) for x in params]
    fit_analysis = K / (1 + np.exp(-r * (x_analysis - x0))) ** (1 / v)
    signal_analysis = normalize_logistic_residual(
        biomass[analysis_idx], fit_analysis, normalization_mode
    )
    smooth_analysis = np.convolve(signal_analysis, kernel, mode="same")

    full_fit = np.full(len(time), np.nan)
    full_signal = np.full(len(time), np.nan)
    full_smooth = np.full(len(time), np.nan)
    full_fit[analysis_idx] = fit_analysis
    full_signal[analysis_idx] = signal_analysis
    full_smooth[analysis_idx] = smooth_analysis

    return {
        "fit": full_fit,
        "signal": full_signal,
        "smooth": full_smooth,
        "r2": r2,
        "degree": None,
        "requested_start_h": float(requested_start),
        "requested_end_h": float(requested_end),
        "fit_start_h": float(time[l_lim]),
        "fit_end_h": float(time[u_lim - 1]),
        "signal_start_h": float(time[analysis_idx[0]]),
        "signal_end_h": float(time[analysis_idx[-1]]),
        "window_scores": pd.DataFrame(rows),
    }


def fourier_transform_interval(
    mean_smooth: np.ndarray,
    kernel_size: int,
    zero_padding: int,
    measurement_interval_s: float,
):
    """
    Calculate the FFT using the selected measurement interval.
    The same calculation can be used for polynomial- and logistic-detrended signals.
    """
    valid = np.asarray(mean_smooth, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) < 3:
        raise ValueError("Not enough finite smoothed-signal points for FFT.")

    signal = np.concatenate((valid, np.zeros(len(valid) * int(zero_padding))))
    T = float(measurement_interval_s) / 3600.0
    N = signal.shape[0]
    N_orig = valid.shape[0]
    yf = rfft(signal)
    xf = rfftfreq(N, T)
    amplitudes = 2 / N_orig * np.abs(yf)

    dominant_idx = int(np.argmax(amplitudes))
    # Stability guard only for the failure case in which DC is selected.
    if xf[dominant_idx] == 0 and len(xf) > 1:
        dominant_idx = int(np.argmax(amplitudes[1:]) + 1)

    dominant_freq = float(xf[dominant_idx])
    if dominant_freq <= 0:
        raise ValueError("FFT did not find a non-zero dominant frequency.")

    dominant_period = 1 / dominant_freq
    dominant_amplitude = float(amplitudes[dominant_idx])
    phase_rad = float(np.angle(yf[dominant_idx]))
    phase_shift_hours = radians_to_hours(phase_rad, period_hours=dominant_period)
    phase_correction = kernel_size * float(measurement_interval_s) / 3600.0 / 2
    phase_shift_hours = phase_shift_hours + phase_correction
    corrected_phase_rad = phase_to_radians(phase_shift_hours, period_hours=dominant_period)
    return {
        "dominant_amplitude": dominant_amplitude,
        "dominant_freq": dominant_freq,
        "dominant_period": dominant_period,
        "phase_shift_hours": float(phase_shift_hours),
        "corrected_phase_rad": float(corrected_phase_rad),
        "frequency": xf,
        "amplitude": amplitudes,
    }


def run_cosine_from_fft(time, smooth, sample_id, fft_result):
    result = cosine_fit(
        time=np.asarray(time, dtype=float),
        mean_smooth=np.asarray(smooth, dtype=float),
        name=sample_id,
        dominant_amplitude=fft_result["dominant_amplitude"],
        dominant_freq=fft_result["dominant_freq"],
        corrected_phase_rad=fft_result["corrected_phase_rad"],
    )

    if isinstance(result, tuple) and len(result) == 6:
        A_fit, wavelength, time_shift, time_shift_rad, y_fit, covariance = result
        return {
            "amplitude": float(A_fit),
            "period_h": float(wavelength),
            "phase_shift_h": float(time_shift),
            "phase_shift_rad": float(time_shift_rad),
            "peak_phase_h": cosine_peak_phase_hours(time_shift, wavelength),
            "curve": np.asarray(y_fit, dtype=float),
            "used_approximation": False,
        }

    return {
        "amplitude": fft_result["dominant_amplitude"],
        "period_h": fft_result["dominant_period"],
        "phase_shift_h": fft_result["phase_shift_hours"],
        "phase_shift_rad": fft_result["corrected_phase_rad"],
        "peak_phase_h": cosine_peak_phase_hours(fft_result["phase_shift_hours"], fft_result["dominant_period"]),
        "curve": np.asarray(result, dtype=float),
        "used_approximation": True,
    }


# ============================================================================
# Persistence, plotting, and GUI v0.3
# ============================================================================

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def effective_measurement_interval_s(time: np.ndarray, settings: AnalysisSettings) -> float:
    """Use the Relative Time column when auto-detection is enabled; otherwise use the manual interval."""
    if settings.auto_measurement_interval:
        time = np.asarray(time, dtype=float)
        diffs = np.diff(time)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            return float(np.median(diffs) * 3600.0)
    return float(settings.measurement_interval_s)


def fft_spectrum(smoothed_signal, zero_padding: int, measurement_interval_s: float):
    """Calculate FFT frequencies, amplitudes, and corresponding periods for plotting."""
    smoothed_signal = np.asarray(smoothed_signal, dtype=float)
    clean_signal = smoothed_signal[np.isfinite(smoothed_signal)]
    if clean_signal.size < 3:
        return np.array([]), np.array([]), np.nan

    sample = np.append(clean_signal, np.repeat(0.0, clean_signal.size * int(zero_padding)))
    T = float(measurement_interval_s) / 3600.0
    N = sample.shape[0]
    yf = rfft(sample)
    xf = rfftfreq(N, T)
    amplitude = 2 / N * np.abs(yf)

    period_time = np.nan
    if amplitude.size > 2 and np.nanmax(amplitude) > 0:
        height_thresh = np.nanmax(amplitude) * 0.9
        peak_idx = find_peaks(amplitude, height=height_thresh)[0]
        peak_idx = peak_idx[xf[peak_idx] > 0] if len(peak_idx) else peak_idx
        if len(peak_idx):
            period_time = float(1 / xf[peak_idx][0])
        else:
            nz = np.flatnonzero(xf > 0)
            if len(nz):
                idx = nz[np.argmax(amplitude[nz])]
                if xf[idx] > 0:
                    period_time = float(1 / xf[idx])
    return xf, amplitude, period_time


def find_signal_extrema(smoothed_signal, distance_between_peaks: int, width_peak: int):
    values = np.asarray(smoothed_signal, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.array([], dtype=int)
    work = values.copy()
    # find_peaks cannot use NaNs reliably. Keep invalid edge areas far below/above signal.
    floor = np.nanmin(work[finite]) - max(1.0, np.nanstd(work[finite]) * 10)
    ceil = np.nanmax(work[finite]) + max(1.0, np.nanstd(work[finite]) * 10)
    for_peaks = work.copy(); for_peaks[~finite] = floor
    for_troughs = -work.copy(); for_troughs[~finite] = -ceil
    peaks = find_peaks(for_peaks, distance=max(1, int(distance_between_peaks)), width=max(1, int(width_peak)))[0]
    troughs = find_peaks(for_troughs, distance=max(1, int(distance_between_peaks)), width=max(1, int(width_peak)))[0]
    extrema = np.unique(np.concatenate([peaks, troughs])).astype(int)
    return extrema[np.isfinite(values[extrema])]


def _valid_xlim(start, end):
    if start is None and end is None:
        return None
    return (start, end)


def apply_time_axis(ax, ps: PlotSettings):
    if ps.x_start_h is not None or ps.x_end_h is not None:
        left, right = ax.get_xlim()
        ax.set_xlim(ps.x_start_h if ps.x_start_h is not None else left,
                    ps.x_end_h if ps.x_end_h is not None else right)

    if ps.show_12h_24h:
        major = max(0.001, float(ps.major_tick_h))
        minor = max(0.001, float(ps.minor_tick_h))
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ax.xaxis.set_minor_formatter(NullFormatter())

    if ps.show_grid:
        ax.grid(True, which="major", axis="both", alpha=ps.grid_alpha, linewidth=0.8)
        if ps.show_12h_24h:
            ax.grid(True, which="minor", axis="x", alpha=ps.minor_grid_alpha, linewidth=0.6)

    end = float(ps.entrainment_end_h or 0)
    light = float(ps.light_hours or 0)
    dark = float(ps.dark_hours or 0)
    cycle = light + dark
    if end > 0 and light > 0 and dark > 0 and cycle > 0:
        start = 0.0
        while start < end:
            dark_start = start + light
            dark_end = min(start + cycle, end)
            if dark_start < end:
                ax.axvspan(dark_start, dark_end, alpha=0.08, color="black", zorder=0)
            start += cycle
        ax.axvline(end, linewidth=0.8, linestyle=":", color="black", alpha=0.55)


def apply_plot_axis(ax, ps: PlotSettings, kind: str):
    ax.tick_params(axis="both", labelsize=max(6, ps.font_size - 1))
    if kind in {"biomass", "signal"}:
        apply_time_axis(ax, ps)
    if kind == "biomass":
        if ps.biomass_y_min is not None or ps.biomass_y_max is not None:
            lo, hi = ax.get_ylim()
            ax.set_ylim(ps.biomass_y_min if ps.biomass_y_min is not None else lo,
                        ps.biomass_y_max if ps.biomass_y_max is not None else hi)
    elif kind == "signal":
        if ps.signal_y_min is not None or ps.signal_y_max is not None:
            lo, hi = ax.get_ylim()
            ax.set_ylim(ps.signal_y_min if ps.signal_y_min is not None else lo,
                        ps.signal_y_max if ps.signal_y_max is not None else hi)
    elif kind == "fft":
        ax.set_xlim(ps.fft_x_min, ps.fft_x_max)
        if ps.fft_y_min is not None or ps.fft_y_max is not None:
            lo, hi = ax.get_ylim()
            ax.set_ylim(ps.fft_y_min if ps.fft_y_min is not None else lo,
                        ps.fft_y_max if ps.fft_y_max is not None else hi)
        if ps.show_grid:
            ax.grid(True, which="major", alpha=ps.grid_alpha, linewidth=0.8)


def place_legend(ax, ps: PlotSettings, **kwargs):
    if ps.legend_below:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=max(1, ps.legend_ncol),
                  frameon=False, fontsize=ps.legend_font_size, **kwargs)
    else:
        ax.legend(frameon=False, fontsize=ps.legend_font_size, ncol=max(1, ps.legend_ncol), **kwargs)


def draw_biomass_fit(ax, result: dict, config: SampleConfig, ps: PlotSettings, title=None, show_legend=True):
    t = np.asarray(result["time"], float)
    biomass = np.asarray(result["biomass"], float)
    fit = np.asarray(result["fit"], float)
    c = config.color
    ax.plot(t, biomass, color=c, linewidth=ps.line_width, linestyle=ps.line_style,
            alpha=ps.biomass_alpha, label="Backscatter")
    mask = np.isfinite(t) & np.isfinite(fit)
    if mask.any():
        # Draw the regression fit using the selected fit line style.
        ax.plot(t[mask], fit[mask], color=ps.fit_color, linewidth=ps.fit_line_width,
                linestyle=ps.fit_line_style, alpha=ps.fit_alpha, label="Fit")
    ax.set_title(title or config.plot_label, fontsize=ps.title_font_size)
    ax.set_xlabel("Time [h]", fontsize=ps.font_size)
    ax.set_ylabel("Backscatter [a.u.]", fontsize=ps.font_size)
    apply_plot_axis(ax, ps, "biomass")
    if show_legend:
        place_legend(ax, ps)


def cosine_display_mask(time: np.ndarray, cosine: np.ndarray, ps: PlotSettings) -> np.ndarray:
    """Return the points where the already-fitted cosine should be displayed.

    This intentionally affects plotting only. The fit is still calculated from the
    full analytical signal and all reported cosine metrics remain unchanged.
    """
    time = np.asarray(time, dtype=float)
    cosine = np.asarray(cosine, dtype=float)
    mask = np.isfinite(time) & np.isfinite(cosine)
    if ps.cosine_plot_start_h is not None:
        mask &= time >= float(ps.cosine_plot_start_h)
    if ps.cosine_plot_end_h is not None:
        mask &= time <= float(ps.cosine_plot_end_h)
    return mask


def draw_visual_only_signal_layers(ax, result: dict, config: SampleConfig, ps: PlotSettings, label_prefix="Visual-only"):
    """Draw the optional second logistic residual without feeding it into analysis.

    A dotted line separates this display-only extension from the analytical signal
    used for FFT, cosine fitting, and peak/trough detection.
    """
    t=np.asarray(result.get("time",[]),float)
    raw=np.asarray(result.get("visual_only_signal",np.full(len(t),np.nan)),float)
    smooth=np.asarray(result.get("visual_only_smooth",np.full(len(t),np.nan)),float)
    if ps.show_raw_signal and len(raw)==len(t) and np.isfinite(raw).any():
        m=np.isfinite(t)&np.isfinite(raw)
        ax.plot(t[m],raw[m],color=config.color,linewidth=max(.7,ps.line_width*.7),linestyle=":",
                alpha=min(.65,max(.25,ps.raw_alpha*1.6)),label=f"{label_prefix} raw signal")
    visual_present = (len(raw)==len(t) and np.isfinite(raw).any()) or (len(smooth)==len(t) and np.isfinite(smooth).any())
    if ps.show_smoothed_signal and len(smooth)==len(t) and np.isfinite(smooth).any():
        m=np.isfinite(t)&np.isfinite(smooth)
        ax.plot(t[m],smooth[m],color=config.color,linewidth=ps.line_width,linestyle=":",
                alpha=min(.75,max(.35,ps.smooth_alpha*.65)),label=f"{label_prefix} smoothed signal")

    if ps.show_visual_only_divider and visual_present:
        divider = result.get("visual_only_display_until_h", np.nan)
        if not np.isfinite(divider):
            primary = np.asarray(result.get("signal", np.full(len(t), np.nan)), float)
            if len(primary) == len(t):
                idx = np.flatnonzero(np.isfinite(primary) & np.isfinite(t))
                if idx.size:
                    divider = float(t[idx[0]])
        if np.isfinite(divider):
            # No legend label: replicate overlays may contain several nearly
            # identical boundaries and should not create duplicate legend items.
            ax.axvline(float(divider), color="black", linewidth=max(0.8, ps.line_width * 0.55),
                       linestyle="--", alpha=0.55, zorder=1)


def draw_signal(ax, result: dict, config: SampleConfig, ps: PlotSettings, title=None, show_legend=True):
    t = np.asarray(result["time"], float)
    signal = np.asarray(result["signal"], float)
    smooth = np.asarray(result["smooth"], float)
    cosine = np.asarray(result.get("cosine_curve", np.full(len(t), np.nan)), float)
    c = config.color
    if ps.show_raw_signal:
        ax.plot(t, signal, color=c, linewidth=max(0.7, ps.line_width * 0.7),
                linestyle=ps.line_style, alpha=ps.raw_alpha, label="Raw signal")
    if ps.show_smoothed_signal:
        ax.plot(t, smooth, color=c, linewidth=ps.line_width, linestyle=ps.line_style,
                alpha=ps.smooth_alpha, label="Smoothed signal")
    draw_visual_only_signal_layers(ax,result,config,ps)
    if ps.show_cosine and np.isfinite(cosine).any():
        mask = cosine_display_mask(t, cosine, ps)
        ax.plot(t[mask], cosine[mask], color=ps.cosine_color, linewidth=ps.fit_line_width,
                linestyle=ps.fit_line_style, alpha=ps.cosine_alpha, label="Cosine fit")
    if ps.show_extrema:
        extrema = find_signal_extrema(smooth, ps.peak_distance, ps.peak_width)
        if len(extrema):
            ax.scatter(t[extrema], smooth[extrema], marker="x", color="black", s=25, zorder=5)
            if ps.annotate_extrema:
                finite = smooth[np.isfinite(smooth)]
                yr = np.nanmax(finite) - np.nanmin(finite) if len(finite) else 0
                offset = 0.05 * yr if yr > 0 else 0.05
                for idx in extrema:
                    ax.text(t[idx], smooth[idx] + offset, f"{t[idx]:.1f}h", ha="center", fontsize=max(6, ps.font_size - 4))
    ax.axhline(0, linewidth=0.7, color="black", alpha=0.35)
    ax.set_title(title or f"{config.plot_label} – signal", fontsize=ps.title_font_size)
    ax.set_xlabel("Time [h]", fontsize=ps.font_size)
    ax.set_ylabel(signal_normalization_label(result.get("signal_normalization", "mean_centered")), fontsize=ps.font_size)
    apply_plot_axis(ax, ps, "signal")
    if show_legend:
        place_legend(ax, ps)


def draw_fft(ax, result: dict, config: SampleConfig, ps: PlotSettings, title=None, show_legend=False):
    freq = np.asarray(result.get("fft_display_frequency", result.get("fft_frequency", [])), float)
    amp = np.asarray(result.get("fft_display_amplitude", result.get("fft_amplitude", [])), float)
    if len(freq):
        ax.plot(freq, amp, color=config.color, linewidth=ps.line_width, linestyle=ps.line_style,
                label=config.plot_label)
    ax.set_title(title or f"{config.plot_label} – Fourier transform", fontsize=ps.title_font_size)
    ax.set_xlabel("Frequency [1/h]", fontsize=ps.font_size)
    ax.set_ylabel("FFT amplitude", fontsize=ps.font_size)
    apply_plot_axis(ax, ps, "fft")
    period = result.get("fft_display_period_h")
    if ps.annotate_fft_period and period is not None and np.isfinite(period):
        ax.text(0.98, 0.90, f"period {period:.2f} h", transform=ax.transAxes, ha="right", fontsize=ps.font_size)
    if show_legend:
        place_legend(ax, ps)


def draw_fft_period(ax, result: dict, config: SampleConfig, ps: PlotSettings, title=None, show_legend=False):
    freq = np.asarray(result.get("fft_display_frequency", result.get("fft_frequency", [])), float)
    amp = np.asarray(result.get("fft_display_amplitude", result.get("fft_amplitude", [])), float)
    mask = np.isfinite(freq) & np.isfinite(amp) & (freq > 0)
    if mask.any():
        period = 1.0 / freq[mask]
        order = np.argsort(period)
        ax.plot(period[order], amp[mask][order], color=config.color, linewidth=ps.line_width,
                linestyle=ps.line_style, label=config.plot_label)
    ax.set_title(title or f"{config.plot_label} – Fourier transform by period", fontsize=ps.title_font_size)
    ax.set_xlabel("Period [h]", fontsize=ps.font_size)
    ax.set_ylabel("FFT amplitude", fontsize=ps.font_size)
    # Circadian-focused but broad enough to show common ultradian/circadian peaks.
    ax.set_xlim(4, 72)
    if ps.show_grid:
        ax.grid(True, alpha=ps.grid_alpha)
    ax.tick_params(axis="both", labelsize=max(6, ps.font_size - 1))
    if show_legend:
        place_legend(ax, ps)


_SAVE_WARNING_KEYS = set()


def _warn_save_once(key: str, message: str):
    if key not in _SAVE_WARNING_KEYS:
        _SAVE_WARNING_KEYS.add(key)
        print(f"WARNING: {message}")


def _save_one_figure(fig, path: Path, ps: PlotSettings) -> bool:
    """Save one figure without letting an optional Matplotlib backend abort the run."""
    try:
        fig.savefig(path, bbox_inches="tight", dpi=ps.dpi)
        return True
    except (ModuleNotFoundError, ImportError) as exc:
        if path.suffix.lower() == ".svg" and "backend_svg" in str(exc):
            _warn_save_once(
                "missing_backend_svg",
                "Matplotlib SVG backend is unavailable. SVG files will be skipped; "
                "PDF output will continue. Re-run the updated launcher once to repair Matplotlib."
            )
            return False
        raise


def save_figure(fig, base_path: Path, ps: PlotSettings):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ps.output_format.lower()
    pdf_saved = False
    if fmt in {"pdf", "both", "all"}:
        pdf_saved = _save_one_figure(fig, base_path.with_suffix(".pdf"), ps)
    if fmt in {"svg", "both", "all"}:
        svg_saved = _save_one_figure(fig, base_path.with_suffix(".svg"), ps)
        # With SVG-only selected, still leave a usable PDF if the SVG backend is broken.
        if not svg_saved and not pdf_saved:
            _save_one_figure(fig, base_path.with_suffix(".pdf"), ps)
    if fmt in {"png", "all"}:
        _save_one_figure(fig, base_path.with_suffix(".png"), ps)
    plt.close(fig)


def save_figure_collection(figs, base_path: Path, ps: PlotSettings):
    """Save a multi-page PDF and/or one SVG/PNG file for each page."""
    if not figs:
        return
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ps.output_format.lower()
    pdf_saved = False
    if fmt in {"pdf", "both", "all"}:
        with PdfPages(base_path.with_suffix(".pdf")) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight", dpi=ps.dpi)
        pdf_saved = True
    if fmt in {"svg", "both", "all"}:
        svg_failed = False
        for i, fig in enumerate(figs, start=1):
            if not _save_one_figure(fig, base_path.parent / f"{base_path.name}_{i:02d}.svg", ps):
                svg_failed = True
                break
        if svg_failed and not pdf_saved:
            with PdfPages(base_path.with_suffix(".pdf")) as pdf:
                for fig in figs:
                    pdf.savefig(fig, bbox_inches="tight", dpi=ps.dpi)
    if fmt in {"png", "all"}:
        for i, fig in enumerate(figs, start=1):
            _save_one_figure(fig, base_path.parent / f"{base_path.name}_{i:02d}.png", ps)
    for fig in figs:
        plt.close(fig)


def new_panel_grid(n: int, ps: PlotSettings, ncols: Optional[int] = None):
    ncols = max(1, int(ncols or ps.overview_ncols))
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ps.figure_width * min(ncols, 3) / 1.8,
                                      ps.figure_height * nrows / 1.15),
                             squeeze=False)
    return fig, axes.flatten(), n


def interpolate_series(time, values, common_time):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(time) & np.isfinite(values)
    if mask.sum() < 2:
        return np.full(len(common_time), np.nan)
    x, y = time[mask], values[mask]
    order = np.argsort(x); x, y = x[order], y[order]
    # Remove duplicate times before np.interp.
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return np.full(len(common_time), np.nan)
    out = np.interp(common_time, x, y)
    out[(common_time < x[0]) | (common_time > x[-1])] = np.nan
    return out


def common_time_for_results(results: List[dict]):
    if not results:
        return np.array([])
    starts = [np.nanmin(r["time"]) for r in results]
    ends = [np.nanmax(r["time"]) for r in results]
    start, end = max(starts), min(ends)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return np.array([])
    steps = []
    for r in results:
        d = np.diff(np.asarray(r["time"], float))
        d = d[np.isfinite(d) & (d > 0)]
        if len(d): steps.append(float(np.median(d)))
    if not steps:
        return np.array([])
    step = float(np.median(steps))
    return np.arange(start, end + step / 2, step)


def build_group_stats(results: List[dict]) -> Optional[pd.DataFrame]:
    common = common_time_for_results(results)
    if len(common) < 2:
        return None
    out = pd.DataFrame({"Time [h]": common})
    for label, key in [
        ("Backscatter", "biomass"), ("Fit", "fit"), ("Signal", "signal"),
        ("Smoothed_Signal", "smooth"), ("Cosine", "cosine_curve")]:
        matrix = np.vstack([interpolate_series(r["time"], r[key], common) for r in results])
        frame = pd.DataFrame(matrix)
        out[f"{label}_Mean"] = frame.mean(axis=0, skipna=True).to_numpy()
        out[f"{label}_Median"] = frame.median(axis=0, skipna=True).to_numpy()
        out[f"{label}_SD"] = frame.std(axis=0, skipna=True, ddof=0).to_numpy()
    return out


def build_group_median(results: List[dict]) -> Optional[pd.DataFrame]:
    stats = build_group_stats(results)
    if stats is None:
        return None
    cols = ["Time [h]", "Backscatter_Median", "Fit_Median", "Signal_Median",
            "Smoothed_Signal_Median", "Cosine_Median"]
    return stats[cols].copy()


def group_configs_results(all_results: Dict[str, dict], configs: List[SampleConfig]):
    groups = defaultdict(list)
    config_map = {c.sample_id: c for c in configs}
    for c in configs:
        if c.sample_id in all_results:
            groups[c.group].append((config_map[c.sample_id], all_results[c.sample_id]))
    return groups


def parse_replicate_number(sample_id: str):
    s = str(sample_id)
    # Detect replicate numbers written at the end of a sample name.
    m = re.match(r"^.+[.-](\d+)$", s)
    if m:
        return int(m.group(1))
    # For names such as sample-1_2, use the first number as the replicate number.
    m = re.match(r"^.+[-_](\d+)_(\d+)$", s)
    if m:
        return int(m.group(1))
    m = re.search(r"rep(?:licate)?[_-]?(\d+)$", s, flags=re.I)
    return int(m.group(1)) if m else None


def _make_mean_result(stats: pd.DataFrame, group: str, interval_s: float, settings: AnalysisSettings):
    t = stats["Time [h]"].to_numpy(float)
    smooth = stats["Smoothed_Signal_Mean"].to_numpy(float)
    xf, amp, period = fft_spectrum(smooth, settings.zero_padding, interval_s)
    return {
        "time": t,
        "biomass": stats["Backscatter_Mean"].to_numpy(float),
        "fit": stats["Fit_Mean"].to_numpy(float),
        "signal": stats["Signal_Mean"].to_numpy(float),
        "smooth": smooth,
        "cosine_curve": stats["Cosine_Mean"].to_numpy(float),
        "fft_display_frequency": xf,
        "fft_display_amplitude": amp,
        "fft_display_period_h": period,
        "signal_normalization": settings.logistic_signal_normalization if settings.method == "logistic" else "mean_centered",
    }


def _make_group_summary_result(stats: pd.DataFrame, group: str, interval_s: float,
                               settings: AnalysisSettings, statistic: str,
                               raw_results: Optional[List[dict]] = None):
    """Create a mean or median group trace with its own FFT for comparison plots."""
    statistic = str(statistic).lower()
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    suffix = "Mean" if statistic == "mean" else "Median"
    t = stats["Time [h]"].to_numpy(float)
    smooth = stats[f"Smoothed_Signal_{suffix}"].to_numpy(float)
    xf, amp, period = fft_spectrum(smooth, settings.zero_padding, interval_s)
    out = {
        "time": t,
        "biomass": stats[f"Backscatter_{suffix}"].to_numpy(float),
        "fit": stats[f"Fit_{suffix}"].to_numpy(float),
        "signal": stats[f"Signal_{suffix}"].to_numpy(float),
        "smooth": smooth,
        "cosine_curve": stats[f"Cosine_{suffix}"].to_numpy(float),
        "fft_display_frequency": xf,
        "fft_display_amplitude": amp,
        "fft_display_period_h": period,
        "signal_normalization": settings.logistic_signal_normalization if settings.method == "logistic" else "mean_centered",
    }
    if raw_results:
        out.update(visual_only_group_summary(raw_results, t, statistic))
        primary_idx = np.flatnonzero(np.isfinite(out["signal"]) & np.isfinite(t))
        out["visual_only_display_until_h"] = float(t[primary_idx[0]]) if primary_idx.size else np.nan
    return out


def plot_individual_results(all_results, configs, root: Path, ps: PlotSettings):
    if not ps.generate_individual:
        return
    for c in configs:
        sid = c.sample_id
        if sid not in all_results:
            continue
        r = all_results[sid]
        od = root / "01_individual" / slugify(sid)
        for name, drawer in [("biomass_fit", draw_biomass_fit), ("signal", draw_signal), ("fft", draw_fft)]:
            fig, ax = plt.subplots(figsize=(ps.figure_width, ps.figure_height))
            drawer(ax, r, c, ps)
            fig.tight_layout()
            save_figure(fig, od / name, ps)
        if ps.generate_fft_period_view:
            fig, ax = plt.subplots(figsize=(ps.figure_width, ps.figure_height))
            draw_fft_period(ax, r, c, ps)
            fig.tight_layout(); save_figure(fig, od / "fft_period", ps)


def plot_sample_overview_grids(all_results, configs, root: Path, ps: PlotSettings):
    if not ps.generate_overview_grids:
        return
    items = [(c, all_results[c.sample_id]) for c in configs if c.sample_id in all_results]
    if not items:
        return
    for filename, drawer, suptitle in [
        ("biomass_fit_overview_all", draw_biomass_fit, "Biomass + fit overview"),
        ("signal_overview_all", draw_signal, "Signal overview"),
        ("fft_overview_all", draw_fft, "FFT / Fourier overview")]:
        fig, axes, n = new_panel_grid(len(items), ps)
        for ax, (c, r) in zip(axes, items):
            drawer(ax, r, c, ps, show_legend=True)
        for ax in axes[n:]: ax.axis("off")
        fig.suptitle(suptitle, fontsize=ps.title_font_size + 1)
        fig.tight_layout(); save_figure(fig, root / "02_overviews" / filename, ps)

    # Create one page per condition with separate panels for its replicates.
    groups = group_configs_results(all_results, configs)
    for plot_name, drawer, label in [
        ("biomass_fit_overview_by_condition", draw_biomass_fit, "Biomass + fit"),
        ("signal_overview_by_condition", draw_signal, "Signal"),
        ("fft_overview_by_condition", draw_fft, "FFT / Fourier")]:
        figs = []
        for group, members in groups.items():
            fig, axes, n = new_panel_grid(len(members), ps)
            for ax, (c, r) in zip(axes, members): drawer(ax, r, c, ps, show_legend=True)
            for ax in axes[n:]: ax.axis("off")
            fig.suptitle(f"{label}: {group}", fontsize=ps.title_font_size + 1)
            fig.tight_layout(); figs.append(fig)
        save_figure_collection(figs, root / "02_overviews" / plot_name, ps)


def _overlay_biomass(ax, members, ps):
    for idx, (c, r) in enumerate(members):
        t = np.asarray(r["time"], float); b = np.asarray(r["biomass"], float); f = np.asarray(r["fit"], float)
        ax.plot(t, b, color=c.color, lw=ps.line_width, ls=ps.line_style, alpha=max(0.20, ps.biomass_alpha * 0.75), label=f"{c.plot_label} biomass")
        m = np.isfinite(f)
        if m.any(): ax.plot(t[m], f[m], color=ps.fit_color, lw=ps.fit_line_width, ls=ps.fit_line_style, alpha=ps.fit_alpha, label=f"{c.plot_label} fit")
    ax.set_xlabel("Time [h]", fontsize=ps.font_size); ax.set_ylabel("Backscatter [a.u.]", fontsize=ps.font_size)
    apply_plot_axis(ax, ps, "biomass"); place_legend(ax, ps)


def _overlay_signal(ax, members, ps):
    for c, r in members:
        t = np.asarray(r["time"], float); s = np.asarray(r["signal"], float); sm = np.asarray(r["smooth"], float)
        if ps.show_raw_signal: ax.plot(t, s, color=c.color, lw=max(.7, ps.line_width*.7), alpha=ps.raw_alpha*.7, label=f"{c.plot_label} signal")
        if ps.show_smoothed_signal: ax.plot(t, sm, color=c.color, lw=ps.line_width, alpha=ps.smooth_alpha, label=f"{c.plot_label} smoothed")
        draw_visual_only_signal_layers(ax,r,c,ps,label_prefix=f"{c.plot_label} visual-only")
    ax.axhline(0, color="black", lw=.7, alpha=.3)
    mode = members[0][1].get("signal_normalization", "mean_centered") if members else "mean_centered"
    ax.set_xlabel("Time [h]", fontsize=ps.font_size); ax.set_ylabel(signal_normalization_label(mode), fontsize=ps.font_size)
    apply_plot_axis(ax, ps, "signal"); place_legend(ax, ps)


def _overlay_fft(ax, members, ps):
    for c, r in members:
        ax.plot(r["fft_display_frequency"], r["fft_display_amplitude"], color=c.color,
                lw=ps.line_width, ls=ps.line_style, alpha=.8, label=c.plot_label)
    ax.set_xlabel("Frequency [1/h]", fontsize=ps.font_size); ax.set_ylabel("FFT amplitude", fontsize=ps.font_size)
    apply_plot_axis(ax, ps, "fft"); place_legend(ax, ps)


def plot_replicate_overlays(all_results, configs, root: Path, ps: PlotSettings):
    if not ps.generate_replicate_overlays:
        return
    groups = group_configs_results(all_results, configs)
    definitions = [
        ("biomass_fit", _overlay_biomass, "Biomass + fit replicate overlay"),
        ("signal", _overlay_signal, "Signal replicate overlay"),
        ("fft", _overlay_fft, "FFT replicate overlay"),
    ]
    for key, drawer, title in definitions:
        figs = []
        for group, members in groups.items():
            fig, ax = plt.subplots(figsize=(ps.figure_width, ps.figure_height))
            drawer(ax, members, ps); ax.set_title(f"{group} (n={len(members)})", fontsize=ps.title_font_size)
            fig.tight_layout(); figs.append(fig)
            # Also store each condition separately for easier editing/reuse.
            fig2, ax2 = plt.subplots(figsize=(ps.figure_width, ps.figure_height))
            drawer(ax2, members, ps); ax2.set_title(f"{group} (n={len(members)})", fontsize=ps.title_font_size)
            fig2.tight_layout(); save_figure(fig2, root / "03_replicate_overlays" / key / slugify(group), ps)
        save_figure_collection(figs, root / "03_replicate_overlays" / f"replicates_overlay_{key}", ps)

        # Create one subplot per condition for the overview.
        fig, axes, n = new_panel_grid(len(groups), ps)
        for ax, (group, members) in zip(axes, groups.items()):
            drawer(ax, members, ps); ax.set_title(f"{group} (n={len(members)})", fontsize=ps.title_font_size)
        for ax in axes[n:]: ax.axis("off")
        fig.suptitle(title + " overview", fontsize=ps.title_font_size + 1)
        fig.tight_layout(); save_figure(fig, root / "03_replicate_overlays" / f"replicates_overlay_{key}_overview", ps)


def visual_only_group_summary(results: List[dict], common_time: np.ndarray, statistic: str = "mean"):
    """Create plot-only group summaries for the optional visual logistic residual.

    This helper is used only while drawing figures. Its output is not written into
    analytical condition-statistics tables and is never passed to FFT/cosine code.
    """
    out={}
    for key in ("visual_only_signal","visual_only_smooth"):
        matrix=np.vstack([interpolate_series(r["time"],r.get(key,np.full(len(r["time"]),np.nan)),common_time) for r in results])
        frame=pd.DataFrame(matrix)
        if statistic=="median":
            values=frame.median(axis=0,skipna=True).to_numpy()
        else:
            values=frame.mean(axis=0,skipna=True).to_numpy()
        # Pandas returns NaN where no visual-only values exist, which is exactly
        # what we want so the dotted display line cannot spill into analysis time.
        out[key]=values
    return out


def plot_condition_means(all_results, configs, root: Path, ps: PlotSettings, settings: AnalysisSettings):
    if not ps.generate_condition_means:
        return {}
    groups = group_configs_results(all_results, configs)
    stats_map = {}
    mean_results = {}
    for group, members in groups.items():
        rs = [r for _, r in members]
        stats = build_group_stats(rs)
        if stats is None: continue
        stats_map[group] = stats
        interval = float(np.median([r["measurement_interval_s"] for r in rs]))
        mean_results[group] = _make_mean_result(stats, group, interval, settings)
        # Add an in-memory display-only mean for the optional second logistic fit.
        # It remains separate from stats and therefore cannot influence analytical
        # exports or the FFT/cosine calculations created by _make_mean_result().
        mean_results[group].update(visual_only_group_summary(rs,stats["Time [h]"].to_numpy(float),"mean"))
        od = root / "04_condition_means" / slugify(group)
        od.mkdir(parents=True, exist_ok=True)
        stats.to_csv(od / "condition_statistics.csv", index=False)

        c = SampleConfig(group, group, group, members[0][0].color)
        mr = mean_results[group]
        fig, ax = plt.subplots(figsize=(ps.figure_width, ps.figure_height))
        t = stats["Time [h]"].to_numpy(float); mean = stats["Backscatter_Mean"].to_numpy(float); sd = stats["Backscatter_SD"].to_numpy(float)
        ax.plot(t, mean, color=c.color, lw=ps.line_width, label="Mean biomass")
        ax.fill_between(t, mean-sd, mean+sd, color=c.color, alpha=.18, label="Biomass ± SD")
        fm = stats["Fit_Mean"].to_numpy(float); m=np.isfinite(fm)
        ax.plot(t[m], fm[m], color=ps.fit_color, lw=ps.fit_line_width, ls=ps.fit_line_style, alpha=ps.fit_alpha, label="Mean fit")
        ax.set_title(f"Condition mean: {group} (n={len(members)})", fontsize=ps.title_font_size)
        ax.set_xlabel("Time [h]", fontsize=ps.font_size); ax.set_ylabel("Backscatter [a.u.]", fontsize=ps.font_size)
        apply_plot_axis(ax, ps, "biomass"); place_legend(ax, ps); fig.tight_layout()
        save_figure(fig, od / "condition_mean_fit", ps)

        # Median plots summarize the same replicate groups without changing the individual results.
        median_result = {
            "time": stats["Time [h]"].to_numpy(float),
            "biomass": stats["Backscatter_Median"].to_numpy(float),
            "fit": stats["Fit_Median"].to_numpy(float),
            "signal": stats["Signal_Median"].to_numpy(float),
            "smooth": stats["Smoothed_Signal_Median"].to_numpy(float),
            "cosine_curve": stats["Cosine_Median"].to_numpy(float),
            "signal_normalization": settings.logistic_signal_normalization if settings.method == "logistic" else "mean_centered",
        }
        median_result.update(visual_only_group_summary(rs,median_result["time"],"median"))
        mx, my, mp = fft_spectrum(median_result["smooth"], settings.zero_padding, interval)
        median_result["fft_display_frequency"] = mx
        median_result["fft_display_amplitude"] = my
        median_result["fft_display_period_h"] = mp
        fig, ax = plt.subplots(figsize=(ps.figure_width, ps.figure_height)); draw_biomass_fit(ax, median_result, c, ps, title=f"{group} – replicate median biomass + fit"); fig.tight_layout(); save_figure(fig, od / "condition_median_fit", ps)
        fig, ax = plt.subplots(figsize=(ps.figure_width, ps.figure_height)); draw_signal(ax, median_result, c, ps, title=f"{group} – replicate median signal"); fig.tight_layout(); save_figure(fig, od / "condition_median_signal", ps)
        fig, ax = plt.subplots(figsize=(ps.figure_width, ps.figure_height)); draw_fft(ax, median_result, c, ps, title=f"{group} – FFT of median smoothed signal"); fig.tight_layout(); save_figure(fig, od / "condition_median_fft", ps)

    if not mean_results: return stats_map

    # Create an overview of condition mean biomass and regression fits.
    fig, axes, n = new_panel_grid(len(mean_results), ps)
    for ax, (group, mr) in zip(axes, mean_results.items()):
        stats = stats_map[group]; c = SampleConfig(group, group, group, groups[group][0][0].color)
        t=stats["Time [h]"].to_numpy(float); mean=stats["Backscatter_Mean"].to_numpy(float); sd=stats["Backscatter_SD"].to_numpy(float)
        ax.plot(t, mean, color=c.color, lw=ps.line_width, label="Mean biomass")
        ax.fill_between(t, mean-sd, mean+sd, color=c.color, alpha=.18, label="± SD")
        fm=stats["Fit_Mean"].to_numpy(float); m=np.isfinite(fm); ax.plot(t[m], fm[m], color=ps.fit_color, lw=ps.fit_line_width, ls=ps.fit_line_style, alpha=ps.fit_alpha, label="Mean fit")
        ax.set_title(f"{group} (n={len(groups[group])})", fontsize=ps.title_font_size); ax.set_xlabel("Time [h]"); ax.set_ylabel("Backscatter [a.u.]")
        apply_plot_axis(ax, ps, "biomass"); place_legend(ax, ps)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Condition mean biomass + fit overview", fontsize=ps.title_font_size+1); fig.tight_layout()
    save_figure(fig, root / "04_condition_means" / "condition_mean_fit_overview", ps)

    # Create an overview of the mean detrended signal for each condition.
    fig, axes, n = new_panel_grid(len(mean_results), ps)
    for ax, (group, mr) in zip(axes, mean_results.items()):
        c=SampleConfig(group,group,group,groups[group][0][0].color); draw_signal(ax,mr,c,ps,title=f"{group} (n={len(groups[group])})",show_legend=True)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Condition mean signal overview", fontsize=ps.title_font_size+1); fig.tight_layout()
    save_figure(fig, root / "04_condition_means" / "condition_mean_signal_overview", ps)
    return stats_map


def _all_conditions_biomass(ax, members_flat, ps, first_only=False):
    seen = set()
    for group, members in members_flat.items():
        chosen = members[:1] if first_only else members
        for c, r in chosen:
            t=np.asarray(r["time"],float); b=np.asarray(r["biomass"],float); f=np.asarray(r["fit"],float)
            lbl_b=f"{group} biomass" if (group,"b") not in seen else "_nolegend_"; seen.add((group,"b"))
            lbl_f=f"{group} fit" if (group,"f") not in seen else "_nolegend_"; seen.add((group,"f"))
            ax.plot(t,b,color=c.color,lw=ps.line_width,ls=ps.line_style,alpha=ps.biomass_alpha,label=lbl_b)
            m=np.isfinite(f); ax.plot(t[m],f[m],color=ps.fit_color,lw=ps.fit_line_width,ls=ps.fit_line_style,alpha=ps.fit_alpha,label=lbl_f)
    ax.set_xlabel("Time [h]",fontsize=ps.font_size); ax.set_ylabel("Backscatter [a.u.]",fontsize=ps.font_size); apply_plot_axis(ax,ps,"biomass"); place_legend(ax,ps)


def _all_conditions_signal(ax, groups, ps, first_only=False):
    seen=set(); visual_seen=set(); divider_seen=set()
    for group,members in groups.items():
        chosen=members[:1] if first_only else members
        for c,r in chosen:
            sm=np.asarray(r["smooth"],float); t=np.asarray(r["time"],float)
            lbl=group if group not in seen else "_nolegend_"; seen.add(group)
            ax.plot(t,sm,color=c.color,lw=ps.line_width,ls=ps.line_style,alpha=ps.smooth_alpha,label=lbl)
            vsm=np.asarray(r.get("visual_only_smooth",np.full(len(t),np.nan)),float)
            if ps.show_smoothed_signal and np.isfinite(vsm).any():
                m=np.isfinite(t)&np.isfinite(vsm)
                vlbl=f"{group} visual-only" if group not in visual_seen else "_nolegend_"; visual_seen.add(group)
                ax.plot(t[m],vsm[m],color=c.color,lw=ps.line_width,ls=":",alpha=min(.75,max(.35,ps.smooth_alpha*.65)),label=vlbl)
                if ps.show_visual_only_divider and group not in divider_seen:
                    divider=r.get("visual_only_display_until_h",np.nan)
                    if not np.isfinite(divider):
                        primary=np.asarray(r.get("signal",np.full(len(t),np.nan)),float)
                        idx=np.flatnonzero(np.isfinite(primary)&np.isfinite(t)) if len(primary)==len(t) else np.array([],dtype=int)
                        if idx.size: divider=float(t[idx[0]])
                    if np.isfinite(divider):
                        ax.axvline(float(divider),color="black",lw=max(.8,ps.line_width*.55),ls="--",alpha=.45,zorder=1)
                        divider_seen.add(group)
    first_result = next((r for members in groups.values() for _, r in members), None)
    mode = first_result.get("signal_normalization", "mean_centered") if first_result else "mean_centered"
    ax.axhline(0,color="black",lw=.7,alpha=.3); ax.set_xlabel("Time [h]",fontsize=ps.font_size); ax.set_ylabel(signal_normalization_label(mode),fontsize=ps.font_size); apply_plot_axis(ax,ps,"signal"); place_legend(ax,ps)


def plot_all_conditions(all_results, configs, root: Path, ps: PlotSettings):
    if not ps.generate_all_conditions:
        return
    groups=group_configs_results(all_results,configs)
    if not groups: return
    fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _all_conditions_biomass(ax,groups,ps,False); ax.set_title("All conditions – biomass and regression fit",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,root/"05_all_conditions"/"all_conditions_biomass_fit",ps)
    fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _all_conditions_signal(ax,groups,ps,False); ax.set_title("All conditions – smoothed signal",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,root/"05_all_conditions"/"all_conditions_smoothed_signal",ps)
    fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _all_conditions_biomass(ax,groups,ps,True); ax.set_title("All conditions – first replicate biomass and fit",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,root/"05_all_conditions"/"all_conditions_biomass_fit_first_replicate",ps)
    fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _all_conditions_signal(ax,groups,ps,True); ax.set_title("All conditions – first replicate smoothed signal",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,root/"05_all_conditions"/"all_conditions_smoothed_signal_first_replicate",ps)


def plot_by_replicate_number(all_results, configs, root: Path, ps: PlotSettings):
    if not ps.generate_by_replicate:
        return
    replicate_groups=defaultdict(list)
    for c in configs:
        if c.sample_id not in all_results: continue
        rep=parse_replicate_number(c.sample_id)
        if rep is not None: replicate_groups[rep].append((c,all_results[c.sample_id]))
    for rep,members in sorted(replicate_groups.items()):
        # Plot conditions that share the same replicate number together.
        fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height))
        for c,r in members:
            t=np.asarray(r["time"],float); b=np.asarray(r["biomass"],float); f=np.asarray(r["fit"],float)
            ax.plot(t,b,color=c.color,lw=ps.line_width,alpha=ps.biomass_alpha,label=f"{c.group} biomass")
            m=np.isfinite(f); ax.plot(t[m],f[m],color=ps.fit_color,lw=ps.fit_line_width,ls=ps.fit_line_style,alpha=ps.fit_alpha,label=f"{c.group} fit")
        ax.set_title(f"Replicate {rep} – biomass + fit",fontsize=ps.title_font_size); ax.set_xlabel("Time [h]"); ax.set_ylabel("Backscatter [a.u.]"); apply_plot_axis(ax,ps,"biomass"); place_legend(ax,ps); fig.tight_layout(); save_figure(fig,root/"06_by_replicate"/f"replicate_{rep}_biomass_fit",ps)
        fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height))
        for c,r in members:
            ax.plot(r["time"],r["smooth"],color=c.color,lw=ps.line_width,label=c.group)
            vsm=np.asarray(r.get("visual_only_smooth",np.full(len(r["time"]),np.nan)),float)
            if np.isfinite(vsm).any():
                m=np.isfinite(vsm); ax.plot(np.asarray(r["time"])[m],vsm[m],color=c.color,lw=ps.line_width,ls=":",alpha=min(.75,max(.35,ps.smooth_alpha*.65)),label=f"{c.group} visual-only")
                if ps.show_visual_only_divider:
                    divider=r.get("visual_only_display_until_h",np.nan)
                    if np.isfinite(divider):
                        ax.axvline(float(divider),color="black",lw=max(.8,ps.line_width*.55),ls="--",alpha=.45,zorder=1)
        mode = members[0][1].get("signal_normalization", "mean_centered") if members else "mean_centered"
        ax.set_title(f"Replicate {rep} – smoothed signal",fontsize=ps.title_font_size); ax.set_xlabel("Time [h]"); ax.set_ylabel(signal_normalization_label(mode)); apply_plot_axis(ax,ps,"signal"); place_legend(ax,ps); fig.tight_layout(); save_figure(fig,root/"06_by_replicate"/f"replicate_{rep}_signal",ps)


def _plot_group_summary_biomass_comparison(ax, summaries, colors, labels, ps: PlotSettings, statistic: str, stats_map=None):
    for group, r in summaries.items():
        t=np.asarray(r["time"],float); b=np.asarray(r["biomass"],float); f=np.asarray(r["fit"],float)
        color=colors[group]
        ax.plot(t,b,color=color,lw=ps.line_width,ls=ps.line_style,alpha=ps.biomass_alpha,label=f"{labels[group]} {statistic} biomass")
        if statistic == "mean" and stats_map and group in stats_map:
            sd=stats_map[group]["Backscatter_SD"].to_numpy(float)
            if len(sd)==len(t):
                ax.fill_between(t,b-sd,b+sd,color=color,alpha=0.12,label=f"{labels[group]} ± SD")
        m=np.isfinite(t)&np.isfinite(f)
        if m.any():
            # Keep group identity visible in comparison fits by using the group
            # color; the fit remains distinguished by its dashed line style.
            ax.plot(t[m],f[m],color=color,lw=ps.fit_line_width,ls=ps.fit_line_style,alpha=ps.fit_alpha,label=f"{labels[group]} {statistic} fit")
    ax.set_xlabel("Time [h]",fontsize=ps.font_size); ax.set_ylabel("Backscatter [a.u.]",fontsize=ps.font_size)
    apply_plot_axis(ax,ps,"biomass"); place_legend(ax,ps)


def _plot_group_summary_signal_comparison(ax, summaries, colors, labels, ps: PlotSettings, statistic: str, stats_map=None):
    for group,r in summaries.items():
        t=np.asarray(r["time"],float); sm=np.asarray(r["smooth"],float); color=colors[group]
        ax.plot(t,sm,color=color,lw=ps.line_width,ls=ps.line_style,alpha=ps.smooth_alpha,label=f"{labels[group]} {statistic} smoothed signal")
        if statistic == "mean" and stats_map and group in stats_map:
            sd=stats_map[group]["Smoothed_Signal_SD"].to_numpy(float)
            if len(sd)==len(t):
                ax.fill_between(t,sm-sd,sm+sd,color=color,alpha=0.12,label=f"{labels[group]} signal ± SD")
        # Draw the optional early visual-only helper and divider using the same
        # display-only logic as individual signal figures.
        c=SampleConfig(group,group,labels[group],color)
        draw_visual_only_signal_layers(ax,r,c,ps,label_prefix=f"{labels[group]} visual-only")
        cosine=np.asarray(r.get("cosine_curve",np.full(len(t),np.nan)),float)
        if ps.show_cosine and np.isfinite(cosine).any():
            m=cosine_display_mask(t,cosine,ps)
            if m.any():
                ax.plot(t[m],cosine[m],color=color,lw=ps.fit_line_width,ls="--",alpha=min(ps.cosine_alpha,.75),label=f"{labels[group]} mean cosine" if statistic=="mean" else f"{labels[group]} median cosine")
    first=next(iter(summaries.values()),None)
    mode=first.get("signal_normalization","mean_centered") if first else "mean_centered"
    ax.axhline(0,color="black",lw=.7,alpha=.3); ax.set_xlabel("Time [h]",fontsize=ps.font_size); ax.set_ylabel(signal_normalization_label(mode),fontsize=ps.font_size)
    apply_plot_axis(ax,ps,"signal"); place_legend(ax,ps)


def _plot_group_summary_fft_comparison(ax, summaries, colors, labels, ps: PlotSettings, statistic: str):
    for group,r in summaries.items():
        ax.plot(r["fft_display_frequency"],r["fft_display_amplitude"],color=colors[group],lw=ps.line_width,ls=ps.line_style,alpha=.9,label=f"{labels[group]} {statistic} FFT")
    ax.set_xlabel("Frequency [1/h]",fontsize=ps.font_size); ax.set_ylabel("FFT amplitude",fontsize=ps.font_size)
    apply_plot_axis(ax,ps,"fft"); place_legend(ax,ps)


def _plot_group_fft_period_comparison(ax, summaries, colors, labels, ps: PlotSettings, statistic: str):
    for group,r in summaries.items():
        f=np.asarray(r["fft_display_frequency"],float); a=np.asarray(r["fft_display_amplitude"],float)
        m=np.isfinite(f)&np.isfinite(a)&(f>0)
        if not m.any():
            continue
        period=1.0/f[m]
        order=np.argsort(period)
        ax.plot(period[order],a[m][order],color=colors[group],lw=ps.line_width,alpha=.9,label=f"{labels[group]} {statistic}")
    ax.set_xlabel("Period [h]",fontsize=ps.font_size); ax.set_ylabel("FFT amplitude",fontsize=ps.font_size)
    if ps.show_grid: ax.grid(True,alpha=ps.grid_alpha)
    ax.tick_params(labelsize=max(6,ps.font_size-1)); place_legend(ax,ps)


def _plot_comparison_metric_distributions(ax, chosen, groups, labels, colors, metric, ylabel, ps):
    valid_groups=[]
    for x,g in enumerate(chosen):
        vals=np.asarray([r.get(metric,np.nan) for _,r in groups[g]],float)
        vals=vals[np.isfinite(vals)]
        if not len(vals):
            continue
        valid_groups.append((x,g,vals))
        # Small deterministic horizontal offsets reveal replicates without adding
        # random jitter that would change between exports.
        offsets=np.linspace(-0.10,0.10,len(vals)) if len(vals)>1 else np.array([0.0])
        ax.scatter(np.full(len(vals),x)+offsets,vals,color=colors[g],s=34,alpha=.70,label="_nolegend_",zorder=3)
        ax.scatter([x],[float(np.mean(vals))],marker="D",s=55,color=colors[g],edgecolors="black",linewidths=.6,label="_nolegend_",zorder=4)
        ax.scatter([x],[float(np.median(vals))],marker="_",s=170,color="black",linewidths=2,label="_nolegend_",zorder=5)
    ax.set_xticks(range(len(chosen))); ax.set_xticklabels([labels[g] for g in chosen],rotation=20,ha="right")
    ax.set_ylabel(ylabel,fontsize=ps.font_size); ax.tick_params(labelsize=max(6,ps.font_size-1))
    if ps.show_grid: ax.grid(True,axis="y",alpha=ps.grid_alpha)
    # Explain symbols directly in a compact legend.
    handles=[Line2D([0],[0],marker="o",linestyle="None",color="black",label="Replicate"),
             Line2D([0],[0],marker="D",linestyle="None",color="black",label="Mean"),
             Line2D([0],[0],marker="_",linestyle="None",color="black",markersize=12,label="Median")]
    ax.legend(handles=handles,fontsize=ps.legend_font_size,frameon=False)


def plot_custom_group_comparisons(all_results, configs, root: Path, ps: PlotSettings, settings: AnalysisSettings, log=print):
    """Create complete comparison plot sets for every explicitly saved group combination.

    Custom comparisons deliberately ignore the main group-selector filter. They use
    the groups stored in each comparison and only respect the per-sample Plot checkbox,
    so a saved comparison cannot silently lose a group when the main plot selection is
    later changed.
    """
    comparisons=tuple(ps.custom_group_comparisons or ())
    if not comparisons:
        return
    groups=group_configs_results(all_results,[c for c in configs if c.include_in_plots])
    available=set(groups)
    for idx,combo in enumerate(comparisons,1):
        chosen=[]
        for g in combo:
            if g in available and g not in chosen:
                chosen.append(g)
        missing=[g for g in combo if g not in available]
        if missing:
            log(f"Comparison {idx}: skipped unavailable/unchecked group(s): {', '.join(missing)}")
        if len(chosen)<2:
            log(f"Comparison {idx}: fewer than two groups with plotted samples remain; no comparison plots generated.")
            continue
        subset={g:groups[g] for g in chosen}
        labels={g:g for g in chosen}
        colors={g:groups[g][0][0].color for g in chosen}
        title=" + ".join(chosen)
        folder=root/"08_custom_group_comparisons"/f"{idx:02d}_{slugify(title)}"
        folder.mkdir(parents=True,exist_ok=True)

        # Replicate-level overlays preserve all individual traces.
        fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _all_conditions_biomass(ax,subset,ps,False)
        ax.set_title(f"Comparison: {title} – replicate biomass + fit",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,folder/"01_replicates_biomass_fit",ps)
        fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _all_conditions_signal(ax,subset,ps,False)
        ax.set_title(f"Comparison: {title} – replicate smoothed signal",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,folder/"02_replicates_smoothed_signal",ps)
        members=[member for g in chosen for member in groups[g]]
        fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _overlay_fft(ax,members,ps)
        ax.set_title(f"Comparison: {title} – replicate FFT",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,folder/"03_replicates_fft",ps)

        stats_map={}; mean_results={}; median_results={}
        for g in chosen:
            rs=[r for _,r in groups[g]]
            stats=build_group_stats(rs)
            if stats is None:
                continue
            stats_map[g]=stats
            interval=float(np.median([r["measurement_interval_s"] for r in rs]))
            mean_results[g]=_make_group_summary_result(stats,g,interval,settings,"mean",rs)
            median_results[g]=_make_group_summary_result(stats,g,interval,settings,"median",rs)
            stats.to_csv(folder/f"statistics_{slugify(g)}.csv",index=False)

        if len(mean_results)>=2:
            for statistic,summaries,prefix in [("mean",mean_results,"04_mean"),("median",median_results,"05_median")]:
                fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _plot_group_summary_biomass_comparison(ax,summaries,colors,labels,ps,statistic,stats_map)
                ax.set_title(f"Comparison: {title} – {statistic} biomass + fit",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,folder/f"{prefix}_biomass_fit",ps)
                fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _plot_group_summary_signal_comparison(ax,summaries,colors,labels,ps,statistic,stats_map)
                ax.set_title(f"Comparison: {title} – {statistic} signal",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,folder/f"{prefix}_signal",ps)
                fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _plot_group_summary_fft_comparison(ax,summaries,colors,labels,ps,statistic)
                ax.set_title(f"Comparison: {title} – FFT of {statistic} signal",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,folder/f"{prefix}_fft",ps)
                fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height)); _plot_group_fft_period_comparison(ax,summaries,colors,labels,ps,statistic)
                ax.set_title(f"Comparison: {title} – FFT period view ({statistic})",fontsize=ps.title_font_size); fig.tight_layout(); save_figure(fig,folder/f"{prefix}_fft_period_view",ps)

        # Additional comparison-friendly biological metric plots. Replicates are
        # shown as points, with a diamond for the arithmetic mean and a horizontal
        # marker for the median. These are descriptive plots only.
        metrics=[("cosine_period_h","Cosine period [h]"),("cosine_amplitude","Cosine amplitude"),("cosine_peak_phase_h","Cosine peak phase [h]"),("r2","Regression R²")]
        for j,(metric,ylabel) in enumerate(metrics,1):
            fig,ax=plt.subplots(figsize=(ps.figure_width,ps.figure_height))
            _plot_comparison_metric_distributions(ax,chosen,groups,labels,colors,metric,ylabel,ps)
            ax.set_title(f"Comparison: {title} – {ylabel}",fontsize=ps.title_font_size)
            fig.tight_layout(); save_figure(fig,folder/f"06_metric_{j:02d}_{slugify(metric)}",ps)

        # Compact machine-readable comparison membership table for traceability.
        rows=[]
        for g in chosen:
            for c,r in groups[g]:
                rows.append({"group":g,"sample_id":c.sample_id,"plot_label":c.plot_label,
                             "regression_r2":r.get("r2",np.nan),"cosine_period_h":r.get("cosine_period_h",np.nan),
                             "cosine_amplitude":r.get("cosine_amplitude",np.nan),"cosine_peak_phase_h":r.get("cosine_peak_phase_h",np.nan),
                             "fft_period_h":r.get("fft_display_period_h",np.nan)})
        pd.DataFrame(rows).to_csv(folder/"comparison_sample_metrics.csv",index=False)

def plot_summary_metrics(summary_df: pd.DataFrame, root: Path, ps: PlotSettings):
    if summary_df.empty: return
    metrics=[("regression_r2","Regression R²"),("cosine_period_h","Cosine period [h]"),("cosine_amplitude","Cosine amplitude"),("cosine_phase_shift_h","Cosine phase shift [h]"),("cosine_peak_phase_h","Cosine peak phase [h from t=0]")]
    for col,title in metrics:
        if col not in summary_df.columns: continue
        df=summary_df[["plot_label","color",col]].dropna()
        if df.empty: continue
        fig,ax=plt.subplots(figsize=(max(ps.figure_width,0.55*len(df)+4),ps.figure_height))
        x=np.arange(len(df)); vals=df[col].to_numpy(float)
        ax.bar(x,vals,color=df["color"].tolist(),alpha=.75)
        ax.set_xticks(x); ax.set_xticklabels(df["plot_label"].tolist(),rotation=45,ha="right",fontsize=max(6,ps.font_size-2)); ax.set_ylabel(title,fontsize=ps.font_size); ax.set_title(title,fontsize=ps.title_font_size)
        if ps.show_grid: ax.grid(True,axis="y",alpha=ps.grid_alpha)
        fig.tight_layout(); save_figure(fig,root/"07_summary_metrics"/slugify(col),ps)


def save_separate_legend(configs: List[SampleConfig], root: Path, ps: PlotSettings):
    if not ps.separate_legend: return
    seen={}
    for c in configs: seen.setdefault(c.group,c.color)
    handles=[Line2D([0],[0],color=color,lw=ps.line_width,label=group) for group,color in seen.items()]
    if not handles: return
    fig=plt.figure(figsize=(max(5,ps.figure_width),max(1.2,0.45*math.ceil(len(handles)/max(1,ps.legend_ncol)))))
    fig.legend(handles=handles,loc="center",ncol=max(1,ps.legend_ncol),frameon=False,fontsize=ps.legend_font_size)
    fig.gca().axis("off"); save_figure(fig,root/"legend_groups",ps)


def save_sample_outputs(sample_dir: Path, result: dict, config: SampleConfig, mirror_excel_dir: Path):
    sample_dir.mkdir(parents=True, exist_ok=True); mirror_excel_dir.mkdir(parents=True, exist_ok=True)
    t=np.asarray(result["time"],float)
    norm_mode=result.get("signal_normalization","mean_centered")
    signal_col,smooth_col=signal_column_names(norm_mode)
    visual_signal_col="Visual-only "+signal_col.lower() if norm_mode=="fitted_biomass" else "Visual-only signal [a.u.]"
    visual_smooth_col="Visual-only "+smooth_col.lower() if norm_mode=="fitted_biomass" else "Visual-only smoothed signal [a.u.]"
    cosine_col="Cosine fit [fraction of fitted biomass]" if norm_mode=="fitted_biomass" else "Cosine fit [a.u.]"
    full=pd.DataFrame({
        "Time [h]":t,
        "Backscatter [a.u.]":result["biomass"],
        "Regression fit [a.u.]":result["fit"],
        signal_col:result["signal"],
        smooth_col:result["smooth"],
        cosine_col:result["cosine_curve"],
        "Visual-only logistic fit [a.u.]":result.get("visual_only_fit",np.full(len(t),np.nan)),
        visual_signal_col:result.get("visual_only_signal",np.full(len(t),np.nan)),
        visual_smooth_col:result.get("visual_only_smooth",np.full(len(t),np.nan)),
        "Fit value present":np.isfinite(result["fit"]),
        "Smoothed value present":np.isfinite(result["smooth"]),
    })
    full[["Time [h]","Backscatter [a.u.]"]].to_csv(sample_dir/"01_raw.csv",index=False)
    full[["Time [h]","Backscatter [a.u.]","Regression fit [a.u.]"]].to_csv(sample_dir/"02_regression_fit.csv",index=False)
    full[["Time [h]",signal_col,smooth_col]].to_csv(sample_dir/"03_signal.csv",index=False)
    if np.isfinite(np.asarray(result.get("visual_only_signal",[]),float)).any():
        full[["Time [h]","Visual-only logistic fit [a.u.]",visual_signal_col,visual_smooth_col]].to_csv(sample_dir/"03b_visual_only_logistic_signal.csv",index=False)
    fft_df=pd.DataFrame({"Frequency [1/h]":result["fft_display_frequency"],"FFT amplitude":result["fft_display_amplitude"]})
    fft_df.to_csv(sample_dir/"04_fourier_transform.csv",index=False)
    full[["Time [h]",smooth_col,cosine_col]].to_csv(sample_dir/"05_cosine_fit.csv",index=False)
    full.to_csv(sample_dir/"full_analysis.csv",index=False)
    if result.get("degree_scores") is not None: result["degree_scores"].to_csv(sample_dir/"polynomial_degree_r2.csv",index=False)
    if result.get("window_scores") is not None: result["window_scores"].to_csv(sample_dir/"logistic_window_r2.csv",index=False)

    summary={
        "sample_id":config.sample_id,"group":config.group,"plot_label":config.plot_label,"color":config.color,"include_in_plots":config.include_in_plots,
        "method":result["method"],"signal_normalization":norm_mode,"regression_r2":result["r2"],"polynomial_degree":result.get("degree"),
        "fit_start_h":result["fit_start_h"],"fit_end_h":result["fit_end_h"],
        "signal_start_h":result.get("signal_start_h",result["fit_start_h"]),"signal_end_h":result.get("signal_end_h",result["fit_end_h"]),
        "visual_only_fit_r2":result.get("visual_only_r2",np.nan),
        "visual_only_fit_start_h":result.get("visual_only_fit_start_h",np.nan),
        "visual_only_fit_end_h":result.get("visual_only_fit_end_h",np.nan),
        "visual_only_display_until_h":result.get("visual_only_display_until_h",np.nan),
        "measurement_interval_s":result["measurement_interval_s"],
        "fft_period_h":result.get("fft_display_period_h"),
        "cosine_amplitude":result["cosine_amplitude"],"cosine_period_h":result["cosine_period_h"],
        "cosine_phase_shift_h":result["cosine_phase_shift_h"],"cosine_phase_shift_rad":result["cosine_phase_shift_rad"],
        "cosine_peak_phase_h":result["cosine_peak_phase_h"],
        "cosine_used_fft_approximation":result["cosine_used_approximation"],
    }
    pd.DataFrame([summary]).to_csv(sample_dir/"summary.csv",index=False)

    # An actual Excel workbook for each sample with every calculated value.
    xlsx=mirror_excel_dir/f"{slugify(config.sample_id)}_analysis.xlsx"
    with pd.ExcelWriter(xlsx,engine="openpyxl") as writer:
        full.to_excel(writer,sheet_name="analysis",index=False)
        fft_df.to_excel(writer,sheet_name="FFT",index=False)
        pd.DataFrame([summary]).to_excel(writer,sheet_name="summary",index=False)
        if result.get("degree_scores") is not None: result["degree_scores"].to_excel(writer,sheet_name="degree_R2",index=False)
        if result.get("window_scores") is not None: result["window_scores"].to_excel(writer,sheet_name="window_R2",index=False)
    # Keep a copy in the sample folder too.
    local_xlsx=sample_dir/f"{slugify(config.sample_id)}_analysis.xlsx"
    if local_xlsx != xlsx:
        import shutil; shutil.copy2(xlsx,local_xlsx)
    return summary



def _format_mean_sd(values, digits=2):
    """Format finite replicate values as mean ± sample SD for the readable summary."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—"
    mean = float(np.mean(arr))
    if arr.size == 1:
        return f"{mean:.{digits}f}"
    sd = float(np.std(arr, ddof=1))
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _signal_amplitude_unit(normalization_mode: str) -> str:
    return "relative" if str(normalization_mode) == "fitted_biomass" else "a.u."


def _clean_group_display_label(group: str, member_configs: List[SampleConfig]) -> str:
    """Use clean plotting labels for a condition when they share a clear common label.

    Plot labels are defined per replicate. If all labels are identical, that label is
    used directly. Labels such as ``WT rep1``, ``WT rep2`` or ``WT_rep1`` are reduced
    to ``WT``. If no unambiguous common plotting label can be found, the explicit
    group name is retained instead of guessing a biological name.
    """
    if not member_configs:
        return str(group)
    labels = [str(c.plot_label).strip() for c in member_configs if str(c.plot_label).strip()]
    if not labels:
        return str(group)
    if len(set(labels)) == 1:
        return labels[0]

    cleaned = []
    for c in member_configs:
        label = str(c.plot_label).strip()
        # Explicit replicate suffixes are safe to remove from biological names.
        base = re.sub(r"(?:[\s_.-]+rep(?:licate)?[\s_.-]*\d+)\s*$", "", label, flags=re.I).strip(" _.-")
        if base == label:
            # Also remove a trailing numeric suffix only when it matches the
            # replicate number inferred from the sample ID and is separated by
            # whitespace, underscore, dot or hyphen. This avoids changing names
            # such as KaiA3.
            rep = parse_replicate_number(c.sample_id)
            if rep is not None:
                base = re.sub(rf"[\s_.-]+{rep}\s*$", "", label).strip(" _.-")
        cleaned.append(base or label)
    if cleaned and len(set(cleaned)) == 1:
        return cleaned[0]
    return str(group)


def _circular_peak_phase_summary(phases_h, periods_h):
    """Return circular mean/SD of peak phase while allowing replicate periods to differ.

    Each peak phase is first expressed as a fraction of its own fitted cosine period.
    Circular statistics are calculated on those cycle angles. The resulting mean and
    SD are converted back to hours using the mean fitted period of the included
    replicates. This prevents phases close to the cycle boundary (e.g. 25 h and 1 h
    in a ~26 h rhythm) from being averaged as if they were far apart.
    """
    phase = pd.to_numeric(pd.Series(phases_h), errors="coerce").to_numpy(float)
    period = pd.to_numeric(pd.Series(periods_h), errors="coerce").to_numpy(float)
    mask = np.isfinite(phase) & np.isfinite(period) & (period > 0)
    if not mask.any():
        return np.nan, np.nan
    phase = phase[mask]
    period = period[mask]
    angles = 2 * np.pi * np.mod(phase / period, 1.0)
    mean_angle = float(circmean(angles, high=2*np.pi, low=0))
    std_angle = float(circstd(angles, high=2*np.pi, low=0)) if len(angles) > 1 else 0.0
    mean_period = float(np.mean(period))
    return (mean_angle / (2*np.pi)) * mean_period, (std_angle / (2*np.pi)) * mean_period


def build_biological_summary_tables(summary_df: pd.DataFrame, sample_configs: List[SampleConfig], plotted_only: bool = True):
    """Create readable group- and replicate-level biological result tables."""
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    config_map = {c.sample_id: c for c in sample_configs}
    data = summary_df.copy()
    data["include_in_plots"] = data["sample_id"].map(lambda sid: bool(config_map.get(sid, SampleConfig(str(sid), str(sid), str(sid), "#1f77b4")).include_in_plots))
    data["plot_label"] = data.apply(
        lambda row: (config_map[row["sample_id"]].plot_label if row["sample_id"] in config_map else row.get("plot_label", row["sample_id"])), axis=1
    )

    selected = data[data["include_in_plots"]].copy() if plotted_only else data.copy()
    group_rows = []
    numeric_rows = []
    for group, gdf in selected.groupby("group", sort=False):
        member_configs = [config_map[sid] for sid in gdf["sample_id"] if sid in config_map]
        display_label = _clean_group_display_label(str(group), member_configs)
        periods = pd.to_numeric(gdf["cosine_period_h"], errors="coerce")
        amps = pd.to_numeric(gdf["cosine_amplitude"], errors="coerce")
        fft_periods = pd.to_numeric(gdf.get("fft_period_h", pd.Series(index=gdf.index, dtype=float)), errors="coerce")
        r2s = pd.to_numeric(gdf["regression_r2"], errors="coerce")
        phase_mean, phase_sd = _circular_peak_phase_summary(gdf["cosine_peak_phase_h"], gdf["cosine_period_h"])
        fallback_n = int(pd.Series(gdf["cosine_used_fft_approximation"]).fillna(False).astype(bool).sum())
        norm_modes = [str(v) for v in gdf["signal_normalization"].dropna().unique()]
        norm_mode = norm_modes[0] if len(norm_modes) == 1 else "mixed"
        amp_unit = _signal_amplitude_unit(norm_mode) if norm_mode != "mixed" else "mixed"
        phase_text = "—" if not np.isfinite(phase_mean) else (f"{phase_mean:.2f}" if len(gdf) == 1 else f"{phase_mean:.2f} ± {phase_sd:.2f}")

        group_rows.append({
            "Condition": display_label,
            "n": int(len(gdf)),
            "Cosine period [h]": _format_mean_sd(periods, 2),
            "Cosine amplitude": _format_mean_sd(amps, 3),
            "Amplitude unit": amp_unit,
            "Peak phase [h]": phase_text,
            "FFT period [h]": _format_mean_sd(fft_periods, 2),
            "Regression R²": _format_mean_sd(r2s, 3),
            "FFT fallback [n]": fallback_n,
            "Normalization": "Residual / fitted biomass" if norm_mode == "fitted_biomass" else ("Mean-centered residual" if norm_mode == "mean_centered" else norm_mode),
        })

        def ms(arr):
            arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0: return (np.nan, np.nan)
            return float(np.mean(arr)), (float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0)
        per_m, per_sd = ms(periods); amp_m, amp_sd = ms(amps); fft_m, fft_sd = ms(fft_periods); r2_m, r2_sd = ms(r2s)
        numeric_rows.append({
            "Condition": display_label, "Internal group": str(group), "n": int(len(gdf)),
            "cosine_period_mean_h": per_m, "cosine_period_sd_h": per_sd,
            "cosine_amplitude_mean": amp_m, "cosine_amplitude_sd": amp_sd, "amplitude_unit": amp_unit,
            "peak_phase_circular_mean_h": phase_mean, "peak_phase_circular_sd_h": phase_sd,
            "fft_period_mean_h": fft_m, "fft_period_sd_h": fft_sd,
            "regression_r2_mean": r2_m, "regression_r2_sd": r2_sd,
            "cosine_fft_fallback_n": fallback_n, "signal_normalization": norm_mode,
        })

    # Use the same clean condition label throughout the replicate sheet. Prefer
    # plot-included labels because those are the labels used in presentation
    # figures; fall back to all labels if a whole condition is unchecked.
    group_display_map = {}
    for group in data["group"].drop_duplicates():
        plotted_cfg = [cc for cc in sample_configs if cc.group == group and cc.include_in_plots]
        all_cfg = [cc for cc in sample_configs if cc.group == group]
        group_display_map[group] = _clean_group_display_label(str(group), plotted_cfg or all_cfg)

    sample_rows = []
    for _, row in data.iterrows():
        sid = row["sample_id"]
        c = config_map.get(sid)
        display_group = group_display_map.get(row["group"], str(row["group"]))
        norm_mode = str(row.get("signal_normalization", "mean_centered"))
        sample_rows.append({
            "Sample": c.plot_label if c else row.get("plot_label", sid),
            "Condition": display_group,
            "Cosine period [h]": row.get("cosine_period_h", np.nan),
            "Cosine amplitude": row.get("cosine_amplitude", np.nan),
            "Amplitude unit": _signal_amplitude_unit(norm_mode),
            "Peak phase [h]": row.get("cosine_peak_phase_h", np.nan),
            "FFT period [h]": row.get("fft_period_h", np.nan),
            "Regression R²": row.get("regression_r2", np.nan),
            "FFT fallback": bool(row.get("cosine_used_fft_approximation", False)),
            "Included in plot summary": bool(c.include_in_plots) if c else bool(row.get("include_in_plots", True)),
            "Internal sample ID": sid,
            "Internal group": row["group"],
        })

    return pd.DataFrame(group_rows), pd.DataFrame(sample_rows), pd.DataFrame(numeric_rows)


def _style_biological_summary_sheet(ws, title: str, note: str, header_row: int = 3):
    """Apply a restrained thesis-friendly style to an openpyxl worksheet."""
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.worksheet.table import Table, TableStyleInfo
    from openpyxl.utils import get_column_letter

    max_col = max(1, ws.max_column)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    ws.cell(1, 1, title)
    ws.cell(1, 1).font = Font(bold=True, size=14, color="FFFFFF")
    ws.cell(1, 1).fill = PatternFill("solid", fgColor="355C7D")
    ws.cell(1, 1).alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 24
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max_col)
    ws.cell(2, 1, note)
    ws.cell(2, 1).font = Font(italic=True, size=10, color="555555")
    ws.cell(2, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 34

    header_fill = PatternFill("solid", fgColor="D9E7F2")
    thin = Side(style="thin", color="C8D0D8")
    for cell in ws[header_row]:
        cell.font = Font(bold=True, color="1F1F1F")
        cell.fill = header_fill
        cell.border = Border(bottom=thin)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[header_row].height = 32

    if ws.max_row > header_row and ws.max_column > 0:
        safe_name = re.sub(r"[^A-Za-z0-9_]", "", ws.title)[:20] + "Table"
        ref = f"A{header_row}:{get_column_letter(ws.max_column)}{ws.max_row}"
        table = Table(displayName=safe_name, ref=ref)
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
        ws.add_table(table)

    ws.freeze_panes = f"A{header_row + 1}"
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(ws.max_column)}{ws.max_row}"
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        values = [str(ws.cell(r, col_idx).value or "") for r in range(header_row, min(ws.max_row, header_row + 80) + 1)]
        width = min(34, max(10, max((len(v) for v in values), default=10) + 2))
        ws.column_dimensions[letter].width = width
    for row in ws.iter_rows(min_row=header_row + 1):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=False)


def save_biological_summary(summary_df: pd.DataFrame, sample_configs: List[SampleConfig], output_dir: Path):
    """Save thesis-friendly biological summaries as CSV and a styled Excel workbook."""
    group_plot, sample_table, group_numeric = build_biological_summary_tables(summary_df, sample_configs, plotted_only=True)
    group_all, _, group_all_numeric = build_biological_summary_tables(summary_df, sample_configs, plotted_only=False)

    group_plot.to_csv(output_dir / "biological_summary_by_group.csv", index=False)
    group_all.to_csv(output_dir / "biological_summary_by_group_all_samples.csv", index=False)
    sample_table.to_csv(output_dir / "biological_summary_by_sample.csv", index=False)

    xlsx_path = output_dir / "biological_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        group_plot.to_excel(writer, sheet_name="Group summary", index=False, startrow=2)
        group_all.to_excel(writer, sheet_name="Group summary all", index=False, startrow=2)
        sample_table.to_excel(writer, sheet_name="Sample results", index=False, startrow=2)
        group_numeric.to_excel(writer, sheet_name="Group numeric", index=False, startrow=2)
        group_all_numeric.to_excel(writer, sheet_name="Group numeric all", index=False, startrow=2)

        _style_biological_summary_sheet(
            writer.book["Group summary"],
            "Biological summary – samples included in plots",
            "One row per condition. Period, amplitude, FFT period and R² are mean ± sample SD across included replicates. Peak phase uses a circular mean ± circular SD. Condition names use the clean plotting labels when an unambiguous common label is available.",
        )
        _style_biological_summary_sheet(
            writer.book["Group summary all"],
            "Biological summary – all analyzed samples",
            "Same biological summary, but includes every successfully analyzed sample regardless of the Plot checkbox.",
        )
        _style_biological_summary_sheet(
            writer.book["Sample results"],
            "Biological results by replicate",
            "The Sample column uses the optional clean Plot label entered in the Samples & groups tab. Internal IDs are retained at the right for traceability.",
        )
        _style_biological_summary_sheet(
            writer.book["Group numeric"],
            "Group summary – numeric values",
            "Machine-readable numeric version of the plotted-sample group summary for further statistics or graphing.",
        )
        _style_biological_summary_sheet(
            writer.book["Group numeric all"],
            "Group summary – numeric values, all samples",
            "Machine-readable numeric version including all analyzed samples.",
        )

    return xlsx_path

def save_wide_regression_table(all_results: Dict[str,dict], output_dir: Path):
    wide=pd.DataFrame()
    for sid,r in all_results.items():
        block=pd.DataFrame({
            f"{sid}_Time_Relative":r["time"],
            f"{sid}_Biomass":r["biomass"],
            f"{sid}_reg_predict":r["fit"],
            f"{sid}_signal":r["signal"],
            f"{sid}_smoothed_signal":r["smooth"],
            f"{sid}_cosine_fit":r["cosine_curve"],
            f"{sid}_visual_only_fit":r.get("visual_only_fit",np.full(len(r["time"]),np.nan)),
            f"{sid}_visual_only_signal":r.get("visual_only_signal",np.full(len(r["time"]),np.nan)),
            f"{sid}_visual_only_smoothed_signal":r.get("visual_only_smooth",np.full(len(r["time"]),np.nan)),
        })
        wide=pd.concat([wide,block],axis=1)
    wide.to_csv(output_dir/"regression_data_all_samples.csv",index=False)
    with pd.ExcelWriter(output_dir/"regression_data_all_samples.xlsx",engine="openpyxl") as writer:
        wide.to_excel(writer,sheet_name="regression_data",index=False)
    return wide


def compute_one_result(combined_df: pd.DataFrame, config: SampleConfig, settings: AnalysisSettings, log=print):
    time,biomass=load_combined_sample(combined_df,config.sample_id)
    if settings.method=="polynomial":
        reg=polynomial_fixed_degree(
            time, biomass,
            settings.analysis_start_h, settings.analysis_end_h,
            settings.kernel_size,
            degree=settings.polynomial_degree,
        )
    else:
        reg=logistic_best_window(time,biomass,settings.logistic_start_min_h,settings.logistic_start_max_h,
                                 settings.logistic_end_min_h,settings.logistic_end_max_h,settings.kernel_size,
                                 settings.logistic_grid_points,settings.smooth_sigma,settings.slope_fraction,
                                 settings.logistic_signal_normalization,settings.logistic_signal_start_h,
                                 settings.logistic_signal_end_h)

    # Optional second logistic fit used ONLY to make otherwise omitted timepoints
    # visible in signal plots. These arrays are never passed to FFT, cosine fitting,
    # extrema analysis, R² selection, summaries, or the normal group statistics.
    visual_fit=np.full(len(time),np.nan)
    visual_signal=np.full(len(time),np.nan)
    visual_smooth=np.full(len(time),np.nan)
    visual_r2=np.nan
    visual_fit_start_h=np.nan
    visual_fit_end_h=np.nan
    visual_display_until_h=np.nan
    if settings.method=="logistic" and settings.logistic_visual_fit_enabled:
        vstart=float(settings.logistic_visual_fit_start_h)
        vend=float(settings.logistic_visual_fit_end_h)
        if vend <= vstart:
            log(f"  visual-only logistic fit skipped: end ({vend:g} h) must be greater than start ({vstart:g} h).")
        else:
            try:
                visual_reg=logistic_best_window(
                    time,biomass,
                    vstart,vstart,vend,vend,
                    settings.kernel_size,1,settings.smooth_sigma,settings.slope_fraction,
                    settings.logistic_signal_normalization,vstart,vend
                )
                visual_fit=np.asarray(visual_reg["fit"],float).copy()
                visual_signal=np.asarray(visual_reg["signal"],float).copy()
                visual_smooth=np.asarray(visual_reg["smooth"],float).copy()

                # IMPORTANT: the visual-only fit window and the visual display
                # window are deliberately separate. The user may fit this second
                # logistic curve over any requested interval, including an interval
                # that extends into the main analytical fit. However, the resulting
                # visual-only residual is displayed ONLY until the first finite
                # timepoint of the real analytical signal. This prevents the dotted
                # helper signal from reappearing later or overlapping the analytical
                # signal while still allowing later data to constrain the visual fit.
                primary_signal=np.asarray(reg["signal"],float)
                primary_indices=np.flatnonzero(np.isfinite(primary_signal))
                if primary_indices.size:
                    real_signal_start_h=float(np.asarray(time,float)[primary_indices[0]])
                    hide_visual=np.asarray(time,float) >= real_signal_start_h
                    visual_fit[hide_visual]=np.nan
                    visual_signal[hide_visual]=np.nan
                    visual_smooth[hide_visual]=np.nan
                else:
                    real_signal_start_h=np.nan

                visual_r2=float(visual_reg["r2"])
                visual_fit_start_h=float(visual_reg["fit_start_h"])
                visual_fit_end_h=float(visual_reg["fit_end_h"])
                visual_display_until_h=real_signal_start_h
            except Exception as exc:
                # Because this feature is explicitly display-only, failure must not
                # invalidate the real analysis.
                log(f"  visual-only logistic fit skipped: {exc}")

    interval_s=effective_measurement_interval_s(time,settings)
    # IMPORTANT: downstream frequency/cosine analysis uses ONLY the primary
    # analytical smoothed signal, never the display-only second fit.
    fft_result=fourier_transform_interval(reg["smooth"],settings.kernel_size,settings.zero_padding,interval_s)
    cosine=run_cosine_from_fft(time,reg["smooth"],config.sample_id,fft_result)
    disp_x,disp_y,disp_period=fft_spectrum(reg["smooth"],settings.zero_padding,interval_s)
    return {
        "sample_id":config.sample_id,"group":config.group,"method":settings.method,"time":time,"biomass":biomass,
        "fit":reg["fit"],"signal":reg["signal"],"smooth":reg["smooth"],"r2":reg["r2"],"degree":reg.get("degree"),
        "signal_normalization":settings.logistic_signal_normalization if settings.method=="logistic" else "mean_centered",
        "fit_start_h":reg["fit_start_h"],"fit_end_h":reg["fit_end_h"],
        "signal_start_h":reg.get("signal_start_h",reg["fit_start_h"]),"signal_end_h":reg.get("signal_end_h",reg["fit_end_h"]),
        "degree_scores":reg.get("degree_scores"),
        "window_scores":reg.get("window_scores"),"measurement_interval_s":interval_s,
        "visual_only_fit":visual_fit,"visual_only_signal":visual_signal,"visual_only_smooth":visual_smooth,
        "visual_only_r2":visual_r2,"visual_only_fit_start_h":visual_fit_start_h,"visual_only_fit_end_h":visual_fit_end_h,
        "visual_only_display_until_h":visual_display_until_h,
        "fft_frequency":fft_result["frequency"],"fft_amplitude":fft_result["amplitude"],
        "fft_display_frequency":disp_x,"fft_display_amplitude":disp_y,"fft_display_period_h":disp_period,
        "cosine_curve":cosine["curve"],"cosine_amplitude":cosine["amplitude"],"cosine_period_h":cosine["period_h"],
        "cosine_phase_shift_h":cosine["phase_shift_h"],"cosine_phase_shift_rad":cosine["phase_shift_rad"],
        "cosine_peak_phase_h":cosine["peak_phase_h"],
        "cosine_used_approximation":cosine["used_approximation"],
    }


def analyze_experiment(combined_data_file: str, experiment_dir: str, experiment_name: str,
                       sample_configs: List[SampleConfig], analysis_settings: AnalysisSettings,
                       plot_settings: PlotSettings, log=print):
    combined_df=pd.read_csv(combined_data_file)
    combined_df=combined_df.loc[:,~combined_df.columns.astype(str).str.startswith("Unnamed:")]
    output_dir=Path(experiment_dir)/"backscatter_analysis_app"
    samples_root=output_dir/"samples"; plots_root=output_dir/"plots"; excel_root=output_dir/"analyzed_excel_exports"
    for p in [output_dir,samples_root,plots_root,excel_root]: p.mkdir(parents=True,exist_ok=True)
    with open(output_dir/"settings.json","w",encoding="utf-8") as fh:
        json.dump({"app_version":APP_VERSION,"experiment_name":experiment_name,"analysis":asdict(analysis_settings),
                   "plotting":asdict(plot_settings),"samples":[asdict(c) for c in sample_configs]},fh,indent=2)

    all_results={}; summaries=[]
    for i,c in enumerate(sample_configs,1):
        log(f"\n[{i}/{len(sample_configs)}] Analyzing {c.sample_id} ({analysis_settings.method})")
        try:
            r=compute_one_result(combined_df,c,analysis_settings,log)
            all_results[c.sample_id]=r
            log(f"  interval: {r['measurement_interval_s']:.4g} s | R²={r['r2']:.6f} | cosine period={r['cosine_period_h']:.3f} h | peak phase={r['cosine_peak_phase_h']:.3f} h")
            summaries.append(save_sample_outputs(samples_root/slugify(c.sample_id),r,c,excel_root))
        except Exception as exc:
            log(f"  ERROR: {exc}")
            log(traceback.format_exc())
    if not all_results: raise RuntimeError("No samples were successfully analyzed.")
    summary_df=pd.DataFrame(summaries); summary_df.to_csv(output_dir/"analysis_summary.csv",index=False)
    biological_summary_path = save_biological_summary(summary_df, sample_configs, output_dir)
    log(f"Biological summary table: {biological_summary_path.name}")
    save_wide_regression_table(all_results,output_dir)

    # Group stats CSV/XLSX are saved independently from plotting.
    groups=group_configs_results(all_results,sample_configs)
    group_stats_root=output_dir/"groups"
    for group,members in groups.items():
        stats=build_group_stats([r for _,r in members])
        if stats is not None:
            gd=group_stats_root/slugify(group); gd.mkdir(parents=True,exist_ok=True)
            stats.to_csv(gd/"condition_statistics.csv",index=False)
            with pd.ExcelWriter(gd/"condition_statistics.xlsx",engine="openpyxl") as writer: stats.to_excel(writer,index=False,sheet_name="statistics")

    plotting_configs=filter_plot_configs(sample_configs,plot_settings)
    # Custom comparisons use their own explicitly stored group combinations. They
    # intentionally do not inherit the main group selector, but they still obey
    # each sample's Plot checkbox so unchecked outliers stay excluded.
    comparison_configs=[c for c in sample_configs if c.include_in_plots]
    selected_ids={c.sample_id for c in plotting_configs}
    log("\nGenerating plots...")
    log(f"Main plot selection: {len(plotting_configs)}/{len(sample_configs)} samples across {len(set(c.group for c in plotting_configs))} groups.")
    if plotting_configs:
        plot_individual_results(all_results,plotting_configs,plots_root,plot_settings)
        plot_sample_overview_grids(all_results,plotting_configs,plots_root,plot_settings)
        plot_replicate_overlays(all_results,plotting_configs,plots_root,plot_settings)
        plot_condition_means(all_results,plotting_configs,plots_root,plot_settings,analysis_settings)
        plot_all_conditions(all_results,plotting_configs,plots_root,plot_settings)
        plot_by_replicate_number(all_results,plotting_configs,plots_root,plot_settings)
        plot_summary_metrics(summary_df[summary_df["sample_id"].isin(selected_ids)].copy(),plots_root,plot_settings)
        save_separate_legend(plotting_configs,plots_root,plot_settings)
    else:
        log("No samples/groups are selected for the main plot sets. Analysis tables were still exported.")

    if plot_settings.custom_group_comparisons:
        log(f"Custom comparisons: {len(plot_settings.custom_group_comparisons)} saved comparison set(s).")
        plot_custom_group_comparisons(all_results,comparison_configs,plots_root,plot_settings,analysis_settings,log)
    log(f"\nFinished. Results saved to: {output_dir}")
    return str(output_dir),summary_df


# ============================================================================
# GUI
# ============================================================================

class ScrollableFrame(ttk.Frame):
    def __init__(self,parent,height=300):
        super().__init__(parent)
        self.canvas=tk.Canvas(self,highlightthickness=0,height=height)
        self.scrollbar=ttk.Scrollbar(self,orient="vertical",command=self.canvas.yview)
        self.inner=ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",lambda e:self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window=self.canvas.create_window((0,0),window=self.inner,anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>",lambda e:self.canvas.itemconfigure(self.window,width=e.width))
        self.canvas.pack(side="left",fill="both",expand=True); self.scrollbar.pack(side="right",fill="y")


class BackscatterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Backscatter Analysis App v{APP_VERSION}")
        self.geometry("1280x900"); self.minsize(1050,760)
        self.experiment_dir=None; self.experiment_name=None; self.combined_data_file=None; self.sample_rows=[]
        self.custom_comparisons=[]
        self.detected_min_time_h=0.0; self.detected_max_time_h=216.0
        self.group_refresh_after_id=None
        self.preview_after_id=None; self.preview_canvas=None; self.preview_result_cache={}
        self._build_ui(); self._register_preview_traces()

    def _build_ui(self):
        header=ttk.Frame(self,padding=10); header.pack(fill="x")
        ttk.Label(header,text="Experiment folder:").pack(side="left")
        self.folder_var=tk.StringVar(); ttk.Entry(header,textvariable=self.folder_var).pack(side="left",fill="x",expand=True,padx=8)
        ttk.Button(header,text="Select folder",command=self.select_folder).pack(side="left")
        ttk.Button(header,text="Reload / combine",command=self.reload_data).pack(side="left",padx=(6,0))
        self.notebook=ttk.Notebook(self); self.notebook.pack(fill="both",expand=True,padx=10,pady=(0,8))
        self.samples_tab=ttk.Frame(self.notebook,padding=10); self.analysis_tab=ttk.Frame(self.notebook,padding=10)
        self.plot_tab=ttk.Frame(self.notebook,padding=10); self.preview_tab=ttk.Frame(self.notebook,padding=10); self.run_tab=ttk.Frame(self.notebook,padding=10)
        for tab,name in [(self.samples_tab,"Samples & groups"),(self.analysis_tab,"Analysis"),(self.plot_tab,"Plotting"),(self.preview_tab,"Live preview"),(self.run_tab,"Run & output")]: self.notebook.add(tab,text=name)
        self._build_samples_tab(); self._build_analysis_tab(); self._build_plot_tab(); self._build_preview_tab(); self._build_run_tab()

    def _build_samples_tab(self):
        ttk.Label(self.samples_tab,text="Samples are detected from the Excel filenames, or from LayoutData descriptions when a multi-well BioLector workbook is loaded. Here you can set the group, plot label, and color for each sample. Uncheck Plot to hide an outlier from figures and plotted group means/medians; the sample will still be analyzed and exported.",wraplength=1100).pack(anchor="w",pady=(0,8))
        header=ttk.Frame(self.samples_tab); header.pack(fill="x")
        for text,width in [("Sample ID",30),("Group",26),("Plot label",30),("Color",12),("Plot",7)]: ttk.Label(header,text=text,width=width).pack(side="left",padx=2)
        self.sample_scroll=ScrollableFrame(self.samples_tab,height=520); self.sample_scroll.pack(fill="both",expand=True,pady=(4,0))

    def _entry(self,parent,label,var,row,col,width=14):
        ttk.Label(parent,text=label).grid(row=row,column=col,sticky="w",padx=4,pady=4)
        ttk.Entry(parent,textvariable=var,width=width).grid(row=row,column=col+1,sticky="w",padx=4,pady=4)

    def _build_analysis_tab(self):
        method=ttk.LabelFrame(self.analysis_tab,text="Detrending method",padding=10); method.pack(fill="x",pady=(0,10))
        self.method_var=tk.StringVar(value="polynomial")
        ttk.Radiobutton(method,text="Polynomial regression",value="polynomial",variable=self.method_var,command=self._update_method_visibility).pack(side="left",padx=(0,20))
        ttk.Radiobutton(method,text="Logistic regression",value="logistic",variable=self.method_var,command=self._update_method_visibility).pack(side="left")
        common=ttk.LabelFrame(self.analysis_tab,text="Common settings",padding=10); common.pack(fill="x",pady=(0,10))
        self.kernel_var=tk.StringVar(value="1200"); self.padding_var=tk.StringVar(value="4"); self.interval_var=tk.StringVar(value="20"); self.auto_interval_var=tk.BooleanVar(value=True)
        self._entry(common,"Smoothing kernel [points]",self.kernel_var,0,0); self._entry(common,"FFT zero padding multiplier",self.padding_var,0,2); self._entry(common,"Manual measurement interval [s]",self.interval_var,1,0)
        ttk.Checkbutton(common,text="Auto-detect FFT measurement interval from Relative Time (recommended)",variable=self.auto_interval_var).grid(row=1,column=2,columnspan=2,sticky="w",padx=4,pady=4)
        ttk.Label(common,text="Auto-detection reads the measurement spacing from the Relative Time column. This keeps the FFT frequency and period correct when the sampling interval differs between experiments.",wraplength=1000).grid(row=2,column=0,columnspan=4,sticky="w",padx=4,pady=(4,0))
        ttk.Label(common,text="Cosine phase output: the fitted model is A·cos(2πft + φ). The app keeps the fitted phase parameter in radians and as a signed phase shift in hours, and also reports Cosine peak phase [h]. Peak phase is the predicted position of a cosine maximum within one fitted cycle: (−phase shift) modulo the fitted period. It is measured from Relative Time = 0 and always lies between 0 h and one fitted period. For biological phase comparisons, Relative Time = 0 should represent the same reference event in every sample (for example the start of constant conditions).",wraplength=1000).grid(row=3,column=0,columnspan=4,sticky="w",padx=4,pady=(6,0))
        self.poly_frame=ttk.LabelFrame(self.analysis_tab,text="Polynomial settings",padding=10); self.poly_frame.pack(fill="x",pady=(0,10))
        self.poly_start_var=tk.StringVar(value="0"); self.poly_end_var=tk.StringVar(value="216"); self.poly_degree_var=tk.StringVar(value="4")
        self._entry(self.poly_frame,"Analysis start [h]",self.poly_start_var,0,0); self._entry(self.poly_frame,"Analysis end [h]",self.poly_end_var,0,2)
        self._entry(self.poly_frame,"Polynomial degree",self.poly_degree_var,1,0)
        ttk.Label(self.poly_frame,text="When an experiment is loaded, the analysis range is set to 0 → the maximum detected Relative Time. The polynomial degree controls how flexible the fitted growth trend can be. Default degree = 4. R² is shown only as a fit-quality value and does not change the selected degree.",wraplength=1000).grid(row=2,column=0,columnspan=4,sticky="w",pady=6)
        self.log_frame=ttk.LabelFrame(self.analysis_tab,text="Logistic settings",padding=10); self.log_frame.pack(fill="x",pady=(0,10))
        self.log_start_min_var=tk.StringVar(value="0"); self.log_start_max_var=tk.StringVar(value="0"); self.log_end_min_var=tk.StringVar(value="216"); self.log_end_max_var=tk.StringVar(value="216"); self.log_signal_start_var=tk.StringVar(value="0"); self.log_signal_end_var=tk.StringVar(value="216"); self.log_grid_var=tk.StringVar(value="7"); self.sigma_var=tk.StringVar(value="2"); self.slope_var=tk.StringVar(value="0.05")
        self.log_visual_enabled_var=tk.BooleanVar(value=False); self.log_visual_start_var=tk.StringVar(value="0"); self.log_visual_end_var=tk.StringVar(value="48")
        self.log_normalization_var=tk.StringVar(value="Mean-centered residual [a.u.]")
        self._entry(self.log_frame,"Fit start search min [h]",self.log_start_min_var,0,0); self._entry(self.log_frame,"Fit start search max [h]",self.log_start_max_var,0,2); self._entry(self.log_frame,"Fit end search min [h]",self.log_end_min_var,1,0); self._entry(self.log_frame,"Fit end search max [h]",self.log_end_max_var,1,2)
        self._entry(self.log_frame,"Signal analysis start [h]",self.log_signal_start_var,2,0); self._entry(self.log_frame,"Signal analysis end [h]",self.log_signal_end_var,2,2)
        ttk.Label(self.log_frame,text="The fit-search window is used only to estimate the logistic growth curve. The fitted curve is then evaluated across the separate signal-analysis window, and measured biomass − fitted biomass is calculated over that full interval. This allows a growth fit to be constrained by a shorter region without shortening the FFT/cosine signal. For example, you can fit 0–84 h but calculate the signal from 3 h to the experiment end. Extrapolation outside the fit window should only be used where the logistic curve still represents the biomass trend.",wraplength=1000).grid(row=3,column=0,columnspan=4,sticky="w",padx=4,pady=(4,8))
        self._entry(self.log_frame,"Grid points per fit boundary",self.log_grid_var,4,0); self._entry(self.log_frame,"Gaussian sigma",self.sigma_var,4,2); self._entry(self.log_frame,"Slope fraction",self.slope_var,5,0)
        ttk.Label(self.log_frame,text="Signal normalization").grid(row=5,column=2,sticky="w",padx=4,pady=4)
        ttk.Combobox(self.log_frame,textvariable=self.log_normalization_var,values=["Mean-centered residual [a.u.]","Residual / fitted biomass [relative]"],state="readonly",width=34).grid(row=5,column=3,sticky="w",padx=4,pady=4)
        ttk.Label(self.log_frame,text="Mean-centered residual calculates measured biomass − logistic fit across the complete signal-analysis window and subtracts the arithmetic mean of that residual. Residual / fitted biomass divides the residual by the fitted biomass, giving a dimensionless fraction. Points where fitted biomass is effectively zero are left undefined. The normalization changes the signal representation, not the logistic fit or its R².",wraplength=1000).grid(row=6,column=0,columnspan=4,sticky="w",padx=4,pady=(4,8))
        ttk.Checkbutton(self.log_frame,text="Add separate display-only logistic fit for omitted timepoints",variable=self.log_visual_enabled_var).grid(row=7,column=0,columnspan=4,sticky="w",padx=4,pady=(8,2))
        self._entry(self.log_frame,"Visual-only fit start [h]",self.log_visual_start_var,8,0); self._entry(self.log_frame,"Visual-only fit end [h]",self.log_visual_end_var,8,2)
        ttk.Label(self.log_frame,text="The display-only fit is separate from the real analysis. Its residual is shown only before the main Signal analysis start and is never used for FFT, cosine fitting, extrema, R² selection, summaries, or group statistics.",wraplength=1000).grid(row=9,column=0,columnspan=4,sticky="w",pady=6)
        ttk.Label(self.log_frame,text="On load, both the fit search and signal-analysis range default to 0 → the maximum detected Relative Time. Widening the fit start/end search ranges selects the best fitting window by R²; it no longer shortens the analytical signal.",wraplength=1000).grid(row=10,column=0,columnspan=4,sticky="w",pady=6)
        self._update_method_visibility()

    def _build_plot_tab(self):
        scroll=ScrollableFrame(self.plot_tab,height=650); scroll.pack(fill="both",expand=True); parent=scroll.inner
        general=ttk.LabelFrame(parent,text="General appearance",padding=10); general.pack(fill="x",pady=(0,8))
        self.format_var=tk.StringVar(value="all"); self.figw_var=tk.StringVar(value="9"); self.figh_var=tk.StringVar(value="4.8"); self.dpi_var=tk.StringVar(value="600"); self.linewidth_var=tk.StringVar(value="2.0"); self.linestyle_var=tk.StringVar(value="-"); self.fitlinewidth_var=tk.StringVar(value="2.0"); self.fitlinestyle_var=tk.StringVar(value="--"); self.fontsize_var=tk.StringVar(value="11"); self.titlefontsize_var=tk.StringVar(value="12"); self.legendfontsize_var=tk.StringVar(value="9"); self.fit_color_var=tk.StringVar(value="#111111"); self.cosine_color_var=tk.StringVar(value="#111111")
        ttk.Label(general,text="Output format").grid(row=0,column=0,sticky="w",padx=4,pady=4); ttk.Combobox(general,textvariable=self.format_var,values=["pdf","svg","png","both","all"],state="readonly",width=12).grid(row=0,column=1,sticky="w")
        self._entry(general,"Figure width [in]",self.figw_var,0,2); self._entry(general,"Figure height [in]",self.figh_var,0,4); self._entry(general,"DPI",self.dpi_var,1,0); self._entry(general,"Data line width",self.linewidth_var,1,2)
        ttk.Label(general,text="Data line style").grid(row=1,column=4,sticky="w",padx=4); ttk.Combobox(general,textvariable=self.linestyle_var,values=["-","--","-.",":"],state="readonly",width=12).grid(row=1,column=5,sticky="w")
        self._entry(general,"Fit line width",self.fitlinewidth_var,2,0); ttk.Label(general,text="Fit line style").grid(row=2,column=2,sticky="w",padx=4); ttk.Combobox(general,textvariable=self.fitlinestyle_var,values=["-","--","-.",":"],state="readonly",width=12).grid(row=2,column=3,sticky="w")
        self._entry(general,"Axis font size",self.fontsize_var,2,4); self._entry(general,"Title font size",self.titlefontsize_var,3,0); self._entry(general,"Legend font size",self.legendfontsize_var,3,2)
        ttk.Label(general,text="Regression fit color").grid(row=3,column=4,sticky="w",padx=4,pady=4)
        self.fit_color_button=tk.Button(general,text=self.fit_color_var.get(),bg=self.fit_color_var.get(),fg="white",width=12,command=lambda:self.choose_plot_style_color(self.fit_color_var,self.fit_color_button,"Choose regression fit color")); self.fit_color_button.grid(row=3,column=5,sticky="w",padx=4,pady=4)
        ttk.Label(general,text="Cosine fit color").grid(row=4,column=0,sticky="w",padx=4,pady=4)
        self.cosine_color_button=tk.Button(general,text=self.cosine_color_var.get(),bg=self.cosine_color_var.get(),fg="white",width=12,command=lambda:self.choose_plot_style_color(self.cosine_color_var,self.cosine_color_button,"Choose cosine fit color")); self.cosine_color_button.grid(row=4,column=1,sticky="w",padx=4,pady=4)
        ttk.Label(general,text="Export quality: PNG files are saved at the selected DPI (default 600 DPI for thesis-quality raster figures). PDF and SVG are vector formats, so lines and text remain sharp when enlarged; DPI mainly affects PNG and any rasterized elements. 'both' keeps the previous PDF + SVG output, while 'all' saves PDF + SVG + PNG.",wraplength=1000).grid(row=5,column=0,columnspan=6,sticky="w",padx=4,pady=(6,2))

        axes=ttk.LabelFrame(parent,text="Axes, FFT focus and grid",padding=10); axes.pack(fill="x",pady=(0,8))
        self.xstart_var=tk.StringVar(value=""); self.xend_var=tk.StringVar(value=""); self.biomass_ymin_var=tk.StringVar(value=""); self.biomass_ymax_var=tk.StringVar(value=""); self.signal_ymin_var=tk.StringVar(value=""); self.signal_ymax_var=tk.StringVar(value=""); self.fftxmin_var=tk.StringVar(value="0"); self.fftxmax_var=tk.StringVar(value="0.3"); self.fftymin_var=tk.StringVar(value=""); self.fftymax_var=tk.StringVar(value=""); self.major_tick_var=tk.StringVar(value="24"); self.minor_tick_var=tk.StringVar(value="12"); self.guides_var=tk.BooleanVar(value=True); self.grid_var=tk.BooleanVar(value=True)
        self.cosine_plot_start_var=tk.StringVar(value=""); self.cosine_plot_end_var=tk.StringVar(value="")
        self._entry(axes,"Time X start [h]",self.xstart_var,0,0); self._entry(axes,"Time X end [h]",self.xend_var,0,2); self._entry(axes,"Cosine plot start [h]",self.cosine_plot_start_var,0,4); self._entry(axes,"Cosine plot end [h]",self.cosine_plot_end_var,1,4); self._entry(axes,"Biomass Y min",self.biomass_ymin_var,1,0); self._entry(axes,"Biomass Y max",self.biomass_ymax_var,1,2); self._entry(axes,"Signal Y min",self.signal_ymin_var,2,0); self._entry(axes,"Signal Y max",self.signal_ymax_var,2,2); self._entry(axes,"FFT X min [1/h]",self.fftxmin_var,3,0); self._entry(axes,"FFT X max [1/h]",self.fftxmax_var,3,2); self._entry(axes,"FFT Y min",self.fftymin_var,4,0); self._entry(axes,"FFT Y max",self.fftymax_var,4,2); self._entry(axes,"Major tick interval [h]",self.major_tick_var,5,0); self._entry(axes,"Minor tick interval [h]",self.minor_tick_var,5,2)
        ttk.Checkbutton(axes,text="Use major/minor time marks",variable=self.guides_var).grid(row=6,column=0,columnspan=2,sticky="w",padx=4); ttk.Checkbutton(axes,text="Show grid",variable=self.grid_var).grid(row=6,column=2,columnspan=2,sticky="w",padx=4)
        ttk.Label(axes,text="Cosine plot start/end are display-only. Leave them blank to draw the fitted cosine over its full available time range. Cropping the plotted cosine does not refit it and does not change period, amplitude, phase, FFT, or summary values. The FFT plot defaults to 0–0.3 1/h to focus the frequency axis on the range relevant for circadian oscillations.",wraplength=1000).grid(row=7,column=0,columnspan=6,sticky="w",padx=4,pady=4)

        layers=ttk.LabelFrame(parent,text="Signal / FFT layers",padding=10); layers.pack(fill="x",pady=(0,8))
        self.show_raw_var=tk.BooleanVar(value=True); self.show_smooth_var=tk.BooleanVar(value=True); self.show_cosine_var=tk.BooleanVar(value=True); self.show_visual_divider_var=tk.BooleanVar(value=True); self.show_extrema_var=tk.BooleanVar(value=True); self.annotate_extrema_var=tk.BooleanVar(value=True); self.annotate_fft_var=tk.BooleanVar(value=True); self.peak_width_var=tk.StringVar(value="100"); self.peak_distance_var=tk.StringVar(value="1000"); self.biomass_alpha_var=tk.StringVar(value="0.65"); self.raw_alpha_var=tk.StringVar(value="0.25"); self.smooth_alpha_var=tk.StringVar(value="0.90"); self.fit_alpha_var=tk.StringVar(value="0.95"); self.cosine_alpha_var=tk.StringVar(value="0.95")
        ttk.Checkbutton(layers,text="Raw signal",variable=self.show_raw_var).grid(row=0,column=0,sticky="w",padx=4); ttk.Checkbutton(layers,text="Smoothed signal",variable=self.show_smooth_var).grid(row=0,column=1,sticky="w",padx=4); ttk.Checkbutton(layers,text="Cosine fit",variable=self.show_cosine_var).grid(row=0,column=2,sticky="w",padx=4); ttk.Checkbutton(layers,text="Dashed divider: visual-only → analytical signal",variable=self.show_visual_divider_var).grid(row=0,column=3,sticky="w",padx=4); ttk.Checkbutton(layers,text="Peak/trough markers",variable=self.show_extrema_var).grid(row=1,column=0,sticky="w",padx=4); ttk.Checkbutton(layers,text="Annotate extrema times",variable=self.annotate_extrema_var).grid(row=1,column=1,sticky="w",padx=4); ttk.Checkbutton(layers,text="Annotate FFT period",variable=self.annotate_fft_var).grid(row=1,column=2,sticky="w",padx=4)
        self._entry(layers,"Peak width [points]",self.peak_width_var,2,0); self._entry(layers,"Peak distance [points]",self.peak_distance_var,2,2)
        self._entry(layers,"Biomass alpha",self.biomass_alpha_var,3,0); self._entry(layers,"Raw signal alpha",self.raw_alpha_var,3,2)
        self._entry(layers,"Smoothed signal alpha",self.smooth_alpha_var,4,0); self._entry(layers,"Regression fit alpha",self.fit_alpha_var,4,2)
        self._entry(layers,"Cosine fit alpha",self.cosine_alpha_var,5,0)
        ttk.Label(layers,text="Default transparency: biomass 0.65, raw signal 0.25, smoothed signal 0.90, fitted curves 0.95. Lower values make a line more transparent.",wraplength=900).grid(row=6,column=0,columnspan=4,sticky="w",padx=4,pady=(4,0))

        entr=ttk.LabelFrame(parent,text="Light / dark entrainment",padding=10); entr.pack(fill="x",pady=(0,8))
        self.entrain_end_var=tk.StringVar(value="0"); self.light_var=tk.StringVar(value="12"); self.dark_var=tk.StringVar(value="12")
        self._entry(entr,"Annotate entrainment until [h] (0=off)",self.entrain_end_var,0,0); self._entry(entr,"Light duration [h]",self.light_var,0,2); self._entry(entr,"Dark duration [h]",self.dark_var,1,0)

        legends=ttk.LabelFrame(parent,text="Legends and overview layout",padding=10); legends.pack(fill="x",pady=(0,8))
        self.legend_var=tk.BooleanVar(value=True); self.legend_below_var=tk.BooleanVar(value=False); self.legend_ncol_var=tk.StringVar(value="2"); self.overview_ncols_var=tk.StringVar(value="3")
        ttk.Checkbutton(legends,text="Save separate group legend",variable=self.legend_var).grid(row=0,column=0,columnspan=2,sticky="w",padx=4); ttk.Checkbutton(legends,text="Place plot legends below axes",variable=self.legend_below_var).grid(row=0,column=2,columnspan=2,sticky="w",padx=4); self._entry(legends,"Legend columns",self.legend_ncol_var,1,0); self._entry(legends,"Overview columns",self.overview_ncols_var,1,2)

        selection=ttk.LabelFrame(parent,text="Which groups / samples contribute to plots",padding=10); selection.pack(fill="x",pady=(0,8))
        ttk.Label(selection,text="Select one or several groups below (Cmd-click on macOS, Ctrl-click on Windows/Linux). Samples unchecked in the Samples tab are excluded even when their group is selected.",wraplength=1000).grid(row=0,column=0,columnspan=4,sticky="w",padx=4,pady=(0,5))
        self.group_listbox=tk.Listbox(selection,selectmode="extended",exportselection=False,height=6,width=48)
        self.group_listbox.grid(row=1,column=0,rowspan=3,columnspan=2,sticky="nsew",padx=4,pady=3)
        self.group_listbox.bind("<<ListboxSelect>>",lambda e:self.schedule_preview())
        ttk.Button(selection,text="Select all groups",command=self.select_all_plot_groups).grid(row=1,column=2,sticky="ew",padx=6,pady=3)
        ttk.Button(selection,text="Deselect all groups",command=self.deselect_all_plot_groups).grid(row=2,column=2,sticky="ew",padx=6,pady=3)
        ttk.Button(selection,text="Refresh groups from Samples tab",command=self.refresh_group_selector).grid(row=3,column=2,sticky="ew",padx=6,pady=3)
        selection.columnconfigure(0,weight=1)

        comparisons=ttk.LabelFrame(parent,text="Optional custom group-comparison plots",padding=10); comparisons.pack(fill="x",pady=(0,8))
        ttk.Label(comparisons,text="Select two or more groups in the group list above, then click Add comparison. Each saved combination creates replicate overlays plus mean and median biomass, signal, FFT, FFT-period views, and descriptive metric plots. Saved comparisons use the groups stored here even if the main group selector is changed later; samples unchecked in the Samples tab remain excluded. Example: WT + ΔkaiC can be plotted separately from WT + ΔkaiABC.",wraplength=1000).grid(row=0,column=0,columnspan=4,sticky="w",padx=4,pady=(0,5))
        self.comparison_listbox=tk.Listbox(comparisons,selectmode="extended",exportselection=False,height=5,width=60)
        self.comparison_listbox.grid(row=1,column=0,rowspan=3,columnspan=2,sticky="nsew",padx=4,pady=3)
        ttk.Button(comparisons,text="Add selected groups as comparison",command=self.add_custom_comparison).grid(row=1,column=2,sticky="ew",padx=6,pady=3)
        ttk.Button(comparisons,text="Remove selected comparison(s)",command=self.remove_custom_comparison).grid(row=2,column=2,sticky="ew",padx=6,pady=3)
        ttk.Button(comparisons,text="Clear comparisons",command=self.clear_custom_comparisons).grid(row=3,column=2,sticky="ew",padx=6,pady=3)
        comparisons.columnconfigure(0,weight=1)

        outputs=ttk.LabelFrame(parent,text="Plot sets to generate",padding=10); outputs.pack(fill="x",pady=(0,8))
        self.gen_individual_var=tk.BooleanVar(value=True); self.gen_overview_var=tk.BooleanVar(value=True); self.gen_replicates_var=tk.BooleanVar(value=True); self.gen_means_var=tk.BooleanVar(value=True); self.gen_all_conditions_var=tk.BooleanVar(value=True); self.gen_by_replicate_var=tk.BooleanVar(value=True); self.gen_fft_period_var=tk.BooleanVar(value=True)
        for i,(txt,var) in enumerate([("Individual plots",self.gen_individual_var),("Overview grids",self.gen_overview_var),("Replicate overlays",self.gen_replicates_var),("Condition mean ± SD",self.gen_means_var),("All conditions together",self.gen_all_conditions_var),("Group by replicate number",self.gen_by_replicate_var),("Additional FFT period view",self.gen_fft_period_var)]): ttk.Checkbutton(outputs,text=txt,variable=var).grid(row=i//3,column=i%3,sticky="w",padx=6,pady=3)

    def _build_preview_tab(self):
        top=ttk.Frame(self.preview_tab); top.pack(fill="x",pady=(0,6))
        ttk.Label(top,text="Sample:").pack(side="left"); self.preview_sample_var=tk.StringVar(); self.preview_sample_combo=ttk.Combobox(top,textvariable=self.preview_sample_var,state="readonly",width=34); self.preview_sample_combo.pack(side="left",padx=5)
        ttk.Label(top,text="Plot:").pack(side="left",padx=(12,0)); self.preview_type_var=tk.StringVar(value="Biomass + fit"); ttk.Combobox(top,textvariable=self.preview_type_var,state="readonly",width=22,values=["Biomass + fit","Signal","FFT frequency","FFT period"]).pack(side="left",padx=5)
        self.auto_preview_var=tk.BooleanVar(value=True); ttk.Checkbutton(top,text="Update automatically",variable=self.auto_preview_var).pack(side="left",padx=8); ttk.Button(top,text="Refresh preview",command=self.refresh_preview).pack(side="left")
        self.preview_info_var=tk.StringVar(value="Load an experiment to preview a sample."); ttk.Label(self.preview_tab,textvariable=self.preview_info_var,wraplength=1100).pack(anchor="w",pady=(0,6))
        self.preview_holder=ttk.Frame(self.preview_tab); self.preview_holder.pack(fill="both",expand=True)

    def _build_run_tab(self):
        top=ttk.Frame(self.run_tab); top.pack(fill="x")
        self.run_button=ttk.Button(top,text="Run full analysis",command=self.run_analysis); self.run_button.pack(side="left")
        ttk.Button(top,text="Open results folder",command=self.open_results_folder).pack(side="left",padx=8); ttk.Button(top,text="Save current settings",command=self.save_settings_only).pack(side="left")
        ttk.Label(self.run_tab,text="All loaded samples are analyzed and exported. A styled biological_summary.xlsx is created with a condition-level overview and replicate-level results using the optional clean Plot labels. The main Group summary follows the Plot checkboxes so presentation outliers can be omitted transparently; a second sheet also reports all analyzed samples.",wraplength=1100).pack(anchor="w",pady=10)
        self.status=tk.Text(self.run_tab,height=30,wrap="word"); self.status.pack(fill="both",expand=True); self.log("Select an experiment folder to begin.")

    def _register_preview_traces(self):
        vars_to_watch=[self.method_var,self.kernel_var,self.padding_var,self.interval_var,self.auto_interval_var,self.poly_start_var,self.poly_end_var,self.poly_degree_var,self.log_start_min_var,self.log_start_max_var,self.log_end_min_var,self.log_end_max_var,self.log_signal_start_var,self.log_signal_end_var,self.log_grid_var,self.sigma_var,self.slope_var,self.log_normalization_var,self.log_visual_enabled_var,self.log_visual_start_var,self.log_visual_end_var,self.figw_var,self.figh_var,self.linewidth_var,self.linestyle_var,self.fitlinewidth_var,self.fitlinestyle_var,self.fontsize_var,self.titlefontsize_var,self.legendfontsize_var,self.xstart_var,self.xend_var,self.cosine_plot_start_var,self.cosine_plot_end_var,self.biomass_ymin_var,self.biomass_ymax_var,self.signal_ymin_var,self.signal_ymax_var,self.fftxmin_var,self.fftxmax_var,self.fftymin_var,self.fftymax_var,self.major_tick_var,self.minor_tick_var,self.guides_var,self.grid_var,self.show_raw_var,self.show_smooth_var,self.show_cosine_var,self.show_visual_divider_var,self.show_extrema_var,self.annotate_extrema_var,self.annotate_fft_var,self.peak_width_var,self.peak_distance_var,self.biomass_alpha_var,self.raw_alpha_var,self.smooth_alpha_var,self.fit_alpha_var,self.cosine_alpha_var,self.fit_color_var,self.cosine_color_var,self.entrain_end_var,self.light_var,self.dark_var,self.legend_below_var,self.legend_ncol_var,self.preview_sample_var,self.preview_type_var]
        for v in vars_to_watch: v.trace_add("write",lambda *_:self.schedule_preview())

    def schedule_preview(self):
        if not hasattr(self,"auto_preview_var") or not self.auto_preview_var.get(): return
        if self.preview_after_id is not None:
            try:self.after_cancel(self.preview_after_id)
            except Exception:pass
        self.preview_after_id=self.after(650,self.refresh_preview)

    def _update_method_visibility(self):
        if not hasattr(self,"log_frame"): return
        self.poly_frame.configure(text="Polynomial settings – active" if self.method_var.get()=="polynomial" else "Polynomial settings – inactive")
        self.log_frame.configure(text="Logistic settings – active" if self.method_var.get()=="logistic" else "Logistic settings – inactive")
        self.schedule_preview()

    def log(self,message):
        if hasattr(self,"status"):
            self.status.insert("end",str(message)+"\n"); self.status.see("end"); self.update_idletasks()

    def select_folder(self):
        folder=filedialog.askdirectory(title="Select experiment folder")
        if folder: self.folder_var.set(folder); self.reload_data()

    def reload_data(self):
        folder=self.folder_var.get().strip()
        if not folder: messagebox.showwarning("No folder","Select the experiment folder first."); return
        try:
            self.log(f"\nLoading experiment: {folder}")
            names,combined,exp_dir,exp_name=combine_data_from_experiment(folder,log=self.log)
            self.experiment_dir=exp_dir; self.experiment_name=exp_name; self.combined_data_file=combined
            self.detected_min_time_h,self.detected_max_time_h=combined_time_bounds(combined)
            max_text=f"{self.detected_max_time_h:.10g}"
            # Full-data defaults: 0 to the experiment's largest detected time point.
            self.poly_start_var.set("0"); self.poly_end_var.set(max_text)
            self.log_start_min_var.set("0"); self.log_start_max_var.set("0")
            self.log_end_min_var.set(max_text); self.log_end_max_var.set(max_text)
            self.log_signal_start_var.set("0"); self.log_signal_end_var.set(max_text)
            # The plot x-range defaults to the full experiment and can be changed manually.
            self.xstart_var.set("0"); self.xend_var.set(max_text)
            # Custom comparisons belong to the loaded experiment, so start fresh
            # when a new folder/data set is loaded.
            self.custom_comparisons.clear(); self.refresh_custom_comparison_listbox()
            self.populate_samples(default_sample_configs(names)); self.preview_sample_combo["values"]=names
            if names: self.preview_sample_var.set(names[0])
            self.preview_result_cache.clear(); self.log(f"Detected {len(names)} samples. Default analysis range: 0–{max_text} h."); self.notebook.select(self.samples_tab); self.schedule_preview()
        except Exception as exc:
            self.log(traceback.format_exc()); messagebox.showerror("Could not read experiment",str(exc))

    def populate_samples(self,configs):
        for child in self.sample_scroll.inner.winfo_children(): child.destroy()
        self.sample_rows.clear()
        for i,c in enumerate(configs):
            ttk.Label(self.sample_scroll.inner,text=c.sample_id,width=30).grid(row=i,column=0,sticky="w",padx=2,pady=3)
            gv=tk.StringVar(value=c.group); lv=tk.StringVar(value=c.plot_label); cv=tk.StringVar(value=c.color); pv=tk.BooleanVar(value=c.include_in_plots)
            ttk.Entry(self.sample_scroll.inner,textvariable=gv,width=26).grid(row=i,column=1,sticky="ew",padx=2,pady=3); ttk.Entry(self.sample_scroll.inner,textvariable=lv,width=30).grid(row=i,column=2,sticky="ew",padx=2,pady=3)
            btn=tk.Button(self.sample_scroll.inner,text=c.color,bg=c.color,width=12,command=lambda idx=i:self.choose_group_color(idx)); btn.grid(row=i,column=3,sticky="w",padx=2,pady=3)
            ttk.Checkbutton(self.sample_scroll.inner,variable=pv,command=self.schedule_preview).grid(row=i,column=4,sticky="w",padx=10,pady=3)
            gv.trace_add("write",lambda *_:(self.schedule_preview(),self.schedule_group_selector_refresh())); lv.trace_add("write",lambda *_:self.schedule_preview())
            self.sample_rows.append({"sample_id":c.sample_id,"group_var":gv,"label_var":lv,"color_var":cv,"plot_var":pv,"color_button":btn})
        for col in range(5): self.sample_scroll.inner.columnconfigure(col,weight=1 if col in (1,2) else 0)
        self.refresh_group_selector()

    def schedule_group_selector_refresh(self):
        if self.group_refresh_after_id is not None:
            try:self.after_cancel(self.group_refresh_after_id)
            except Exception:pass
        self.group_refresh_after_id=self.after(500,self.refresh_group_selector)

    def refresh_group_selector(self):
        self.group_refresh_after_id=None
        if not hasattr(self,"group_listbox"):
            return
        old_values=list(self.group_listbox.get(0,"end"))
        old_selected={old_values[i] for i in self.group_listbox.curselection() if i < len(old_values)}
        old_all=bool(old_values) and len(old_selected)==len(old_values)
        groups=[]
        for row in self.sample_rows:
            g=row["group_var"].get().strip() or row["sample_id"]
            if g not in groups: groups.append(g)
        self.group_listbox.delete(0,"end")
        for g in groups: self.group_listbox.insert("end",g)
        # First load and an existing all-selected state both mean select all.
        if not old_values or old_all:
            if groups:self.group_listbox.selection_set(0,"end")
        else:
            for i,g in enumerate(groups):
                if g in old_selected:self.group_listbox.selection_set(i)
        available=set(groups)
        cleaned=[]
        for combo in self.custom_comparisons:
            kept=tuple(g for g in combo if g in available)
            if len(kept)>=2 and kept not in cleaned:
                cleaned.append(kept)
        self.custom_comparisons=cleaned
        self.refresh_custom_comparison_listbox()

    def select_all_plot_groups(self):
        if hasattr(self,"group_listbox") and self.group_listbox.size():
            self.group_listbox.selection_set(0,"end")
            self.schedule_preview()

    def deselect_all_plot_groups(self):
        if hasattr(self,"group_listbox"):
            self.group_listbox.selection_clear(0,"end")
            self.schedule_preview()

    def selected_plot_groups(self):
        if not hasattr(self,"group_listbox"):
            return ()
        return tuple(self.group_listbox.get(i) for i in self.group_listbox.curselection())

    def refresh_custom_comparison_listbox(self):
        if not hasattr(self,"comparison_listbox"):
            return
        self.comparison_listbox.delete(0,"end")
        for combo in self.custom_comparisons:
            self.comparison_listbox.insert("end"," + ".join(combo))

    def add_custom_comparison(self):
        combo=tuple(self.selected_plot_groups())
        if len(combo)<2:
            messagebox.showwarning("Choose groups","Select at least two groups in the group list above, then add the comparison.")
            return
        if combo not in self.custom_comparisons:
            self.custom_comparisons.append(combo)
            self.refresh_custom_comparison_listbox()
        self.schedule_preview()

    def remove_custom_comparison(self):
        if not hasattr(self,"comparison_listbox"):
            return
        selected=list(self.comparison_listbox.curselection())
        for idx in reversed(selected):
            if 0<=idx<len(self.custom_comparisons):
                del self.custom_comparisons[idx]
        self.refresh_custom_comparison_listbox()
        self.schedule_preview()

    def clear_custom_comparisons(self):
        self.custom_comparisons.clear()
        self.refresh_custom_comparison_listbox()
        self.schedule_preview()

    def choose_group_color(self,row_index):
        row=self.sample_rows[row_index]; chosen=colorchooser.askcolor(color=row["color_var"].get(),title="Choose group color")[1]
        if not chosen:return
        target=row["group_var"].get().strip()
        for other in self.sample_rows:
            if other["group_var"].get().strip()==target: other["color_var"].set(chosen); other["color_button"].configure(text=chosen,bg=chosen)
        self.schedule_preview()

    def choose_plot_style_color(self, variable, button, title):
        chosen=colorchooser.askcolor(color=variable.get(),title=title)[1]
        if not chosen:
            return
        variable.set(chosen)
        try:
            r,g,b=mpl_colors.to_rgb(chosen); luminance=0.2126*r+0.7152*g+0.0722*b
            fg="black" if luminance>0.55 else "white"
        except Exception:
            fg="white"
        button.configure(text=chosen,bg=chosen,fg=fg)
        self.schedule_preview()

    def get_sample_configs(self):
        return [SampleConfig(r["sample_id"],r["group_var"].get().strip() or r["sample_id"],r["label_var"].get().strip() or r["sample_id"],r["color_var"].get().strip() or "#1f77b4",bool(r["plot_var"].get())) for r in self.sample_rows]

    def get_analysis_settings(self):
        return AnalysisSettings(
            method=self.method_var.get(),
            analysis_start_h=float(safe_float(self.poly_start_var.get(),0)),
            analysis_end_h=float(safe_float(self.poly_end_var.get(),216)),
            polynomial_degree=max(1,safe_int(self.poly_degree_var.get(),4)),
            logistic_start_min_h=float(safe_float(self.log_start_min_var.get(),0)),
            logistic_start_max_h=float(safe_float(self.log_start_max_var.get(),safe_float(self.log_start_min_var.get(),0))),
            logistic_end_min_h=float(safe_float(self.log_end_min_var.get(),216)),
            logistic_end_max_h=float(safe_float(self.log_end_max_var.get(),safe_float(self.log_end_min_var.get(),216))),
            logistic_signal_start_h=float(safe_float(self.log_signal_start_var.get(),0)),
            logistic_signal_end_h=float(safe_float(self.log_signal_end_var.get(),self.detected_max_time_h)),
            logistic_grid_points=safe_int(self.log_grid_var.get(),7),
            logistic_signal_normalization="fitted_biomass" if self.log_normalization_var.get().startswith("Residual / fitted biomass") else "mean_centered",
            logistic_visual_fit_enabled=bool(self.log_visual_enabled_var.get()),
            logistic_visual_fit_start_h=float(safe_float(self.log_visual_start_var.get(),0)),
            logistic_visual_fit_end_h=float(safe_float(self.log_visual_end_var.get(),48)),
            kernel_size=safe_int(self.kernel_var.get(),1200),
            zero_padding=safe_int(self.padding_var.get(),4),
            measurement_interval_s=float(safe_float(self.interval_var.get(),20)),
            auto_measurement_interval=bool(self.auto_interval_var.get()),
            smooth_sigma=float(safe_float(self.sigma_var.get(),2)),
            slope_fraction=float(safe_float(self.slope_var.get(),.05)),
        )

    def get_plot_settings(self):
        return PlotSettings(
            output_format=self.format_var.get(),
            figure_width=float(safe_float(self.figw_var.get(),9)),
            figure_height=float(safe_float(self.figh_var.get(),4.8)),
            dpi=safe_int(self.dpi_var.get(),600),
            line_width=float(safe_float(self.linewidth_var.get(),2)),
            line_style=self.linestyle_var.get(),
            fit_line_width=float(safe_float(self.fitlinewidth_var.get(),2)),
            fit_line_style=self.fitlinestyle_var.get(),
            biomass_alpha=float(safe_float(self.biomass_alpha_var.get(),.65)),
            raw_alpha=float(safe_float(self.raw_alpha_var.get(),.25)),
            fit_alpha=float(safe_float(self.fit_alpha_var.get(),.95)),
            smooth_alpha=float(safe_float(self.smooth_alpha_var.get(),.90)),
            cosine_alpha=float(safe_float(self.cosine_alpha_var.get(),.95)),
            fit_color=self.fit_color_var.get().strip() or "#111111",
            cosine_color=self.cosine_color_var.get().strip() or "#111111",
            font_size=float(safe_float(self.fontsize_var.get(),11)),
            title_font_size=float(safe_float(self.titlefontsize_var.get(),12)),
            legend_font_size=float(safe_float(self.legendfontsize_var.get(),9)),
            x_start_h=safe_float(self.xstart_var.get(),None),
            x_end_h=safe_float(self.xend_var.get(),None),
            biomass_y_min=safe_float(self.biomass_ymin_var.get(),None),
            biomass_y_max=safe_float(self.biomass_ymax_var.get(),None),
            signal_y_min=safe_float(self.signal_ymin_var.get(),None),
            signal_y_max=safe_float(self.signal_ymax_var.get(),None),
            fft_x_min=float(safe_float(self.fftxmin_var.get(),0)),
            fft_x_max=float(safe_float(self.fftxmax_var.get(),.3)),
            fft_y_min=safe_float(self.fftymin_var.get(),None),
            fft_y_max=safe_float(self.fftymax_var.get(),None),
            show_12h_24h=bool(self.guides_var.get()),
            major_tick_h=float(safe_float(self.major_tick_var.get(),24)),
            minor_tick_h=float(safe_float(self.minor_tick_var.get(),12)),
            show_grid=bool(self.grid_var.get()),
            entrainment_end_h=float(safe_float(self.entrain_end_var.get(),0)),
            light_hours=float(safe_float(self.light_var.get(),12)),
            dark_hours=float(safe_float(self.dark_var.get(),12)),
            show_raw_signal=bool(self.show_raw_var.get()),
            show_smoothed_signal=bool(self.show_smooth_var.get()),
            show_cosine=bool(self.show_cosine_var.get()),
            cosine_plot_start_h=safe_float(self.cosine_plot_start_var.get(),None),
            cosine_plot_end_h=safe_float(self.cosine_plot_end_var.get(),None),
            show_visual_only_divider=bool(self.show_visual_divider_var.get()),
            show_extrema=bool(self.show_extrema_var.get()),
            annotate_extrema=bool(self.annotate_extrema_var.get()),
            peak_width=safe_int(self.peak_width_var.get(),100),
            peak_distance=safe_int(self.peak_distance_var.get(),1000),
            annotate_fft_period=bool(self.annotate_fft_var.get()),
            legend_below=bool(self.legend_below_var.get()),
            legend_ncol=safe_int(self.legend_ncol_var.get(),2),
            separate_legend=bool(self.legend_var.get()),
            generate_individual=bool(self.gen_individual_var.get()),
            generate_overview_grids=bool(self.gen_overview_var.get()),
            generate_replicate_overlays=bool(self.gen_replicates_var.get()),
            generate_condition_means=bool(self.gen_means_var.get()),
            generate_all_conditions=bool(self.gen_all_conditions_var.get()),
            generate_by_replicate=bool(self.gen_by_replicate_var.get()),
            overview_ncols=safe_int(self.overview_ncols_var.get(),3),
            generate_fft_period_view=bool(self.gen_fft_period_var.get()),
            selected_groups=self.selected_plot_groups(),
            custom_group_comparisons=tuple(tuple(c) for c in self.custom_comparisons),
        )

    def validate_ready(self):
        if not self.experiment_dir or not self.combined_data_file or not self.experiment_name: raise RuntimeError("Select/reload an experiment folder first.")
        if not self.sample_rows: raise RuntimeError("No samples are loaded.")

    def refresh_preview(self):
        self.preview_after_id=None
        if not self.combined_data_file or not self.preview_sample_var.get(): return
        try:
            sid=self.preview_sample_var.get(); configs={c.sample_id:c for c in self.get_sample_configs()}; c=configs[sid]
            settings=self.get_analysis_settings(); ps=self.get_plot_settings(); combined=pd.read_csv(self.combined_data_file)
            # Include the analysis settings in the cache key so the preview updates after fit-setting changes.
            key=(sid,json.dumps(asdict(settings),sort_keys=True))
            if key not in self.preview_result_cache:
                self.preview_info_var.set(f"Calculating preview for {sid}..."); self.update_idletasks(); self.preview_result_cache={key:compute_one_result(combined,c,settings)}
            r=self.preview_result_cache[key]
            fig=Figure(figsize=(ps.figure_width,ps.figure_height),dpi=100); ax=fig.add_subplot(111)
            kind=self.preview_type_var.get()
            if kind=="Biomass + fit": draw_biomass_fit(ax,r,c,ps)
            elif kind=="Signal": draw_signal(ax,r,c,ps)
            elif kind=="FFT period": draw_fft_period(ax,r,c,ps)
            else: draw_fft(ax,r,c,ps)
            fig.tight_layout()
            if self.preview_canvas is not None:
                self.preview_canvas.get_tk_widget().destroy()
            self.preview_canvas=FigureCanvasTkAgg(fig,master=self.preview_holder); self.preview_canvas.draw(); self.preview_canvas.get_tk_widget().pack(fill="both",expand=True)
            self.preview_info_var.set(f"{sid}: R²={r['r2']:.4f}, detected interval={r['measurement_interval_s']:.3g} s, FFT period={r['fft_display_period_h']:.2f} h, cosine period={r['cosine_period_h']:.2f} h, peak phase={r['cosine_peak_phase_h']:.2f} h")
        except Exception as exc:
            self.preview_info_var.set(f"Preview error: {exc}")

    def save_settings_only(self):
        try:
            self.validate_ready(); out=Path(self.experiment_dir)/"backscatter_analysis_app"; out.mkdir(exist_ok=True)
            payload={"app_version":APP_VERSION,"experiment_name":self.experiment_name,"analysis":asdict(self.get_analysis_settings()),"plotting":asdict(self.get_plot_settings()),"samples":[asdict(c) for c in self.get_sample_configs()]}
            with open(out/"settings.json","w",encoding="utf-8") as fh: json.dump(payload,fh,indent=2)
            self.log(f"Settings saved to {out/'settings.json'}")
        except Exception as exc: messagebox.showerror("Could not save settings",str(exc))

    def run_analysis(self):
        try:
            self.validate_ready(); self.run_button.configure(state="disabled"); self.notebook.select(self.run_tab); self.log("\n=== Starting full analysis ===")
            out,summary=analyze_experiment(self.combined_data_file,self.experiment_dir,self.experiment_name,self.get_sample_configs(),self.get_analysis_settings(),self.get_plot_settings(),log=self.log)
            self.log(f"Successful samples: {len(summary)}"); messagebox.showinfo("Analysis complete",f"Results saved to:\n{out}")
        except Exception as exc:
            self.log("\nERROR:\n"+traceback.format_exc()); messagebox.showerror("Analysis failed",str(exc))
        finally:self.run_button.configure(state="normal")

    def open_results_folder(self):
        if not self.experiment_dir: messagebox.showwarning("No experiment","Select an experiment folder first."); return
        path=Path(self.experiment_dir)/"backscatter_analysis_app"; path.mkdir(exist_ok=True)
        try:
            if sys.platform=="darwin": subprocess.Popen(["open",str(path)])
            elif os.name=="nt": os.startfile(str(path))
            else: subprocess.Popen(["xdg-open",str(path)])
        except Exception as exc: messagebox.showerror("Could not open folder",str(exc))


def main():
    app=BackscatterApp(); app.mainloop()


if __name__=="__main__":
    main()
