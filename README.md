# Backscatter Analysis App v0.3.12 — User Manual

This app is build on the work of https://github.com/rahilgholami/Circadian-rhythmicity (Berwanger et al., 2025), DOI: 10.1093/pnasnexus/pgaf120

This desktop application analyzes backscatter time-series data with either polynomial or logistic detrending. It separates the slowly changing biomass/growth trend from the oscillating residual signal, smooths the residual, performs FFT/Fourier analysis, fits a cosine, calculates biological summary values, and creates publication-ready plots and tables.

The application is intended for BioLector-style backscatter time series but can also use already-combined data in the expected format.


The app can only be used with proper handling and stress testing of the user might lead to unintended results or errors.

---

## 1. Quick start

1. Open the app and select an experiment folder.
2. Click **Reload / combine**.
3. In **Samples & groups**, check the detected samples, group names, plot labels, colors, and the **Plot** checkbox.
4. In **Analysis**, choose **Polynomial regression** or **Logistic regression** and set the analysis parameters.
5. In **Plotting**, choose axis ranges, line appearance, group selection, and optional group comparisons.
6. Use **Live preview** to inspect a sample before running the full experiment.
7. In **Run & output**, start the analysis.
8. Results are written to:

```text
ExperimentName/
    backscatter_analysis_app/
```

---

# 2. Input data and experiment folders

## Typical BioLector experiment structure

```text
ExperimentName/
    attachments/
    dotsdata exports/
    excel exports/
        ExperimentName_sample-1.xlsx
        ExperimentName_sample-2.xlsx
```

Select either the experiment folder or its `excel exports` folder.

For ordinary Excel exports, the app looks for a sheet whose name contains `backscatter` and reads the relative time and raw backscatter columns. Relative time is converted to hours internally.

The importer also supports multi-well BioLector workbooks containing `LayoutData` and `Channel1`. In these files, the sample description is read from `LayoutData`, only raw Channel1 values are used, and the original `Time[h]` axis is retained.

Importing does not automatically shift the time axis, normalize biomass, detrend the signal, or shorten the experiment.

---

# 3. Samples & groups tab

Each detected sample has the following fields.

## Sample ID

The internal sample identifier detected from the source file or BioLector layout. It is used for traceability and output filenames.

## Group

Defines which biological condition the sample belongs to. Replicates that should be averaged or compared as one condition must have exactly the same group name.

Examples:

- `WT`
- `ΔkaiC`
- `Treatment 1`

## Plot label

A clean display name used in plots and biological summary tables. This can be shorter or more publication-friendly than the internal sample ID.

## Color

The plotting color for the sample/group. Changing the color of one sample normally updates other samples with the same group.

## Plot checkbox

Controls whether the sample contributes to figures and plotted group summaries.

If unchecked, the sample is still analyzed and exported, but it is excluded from plotted means/medians, overlays, comparison plots, and plot-based summary figures. This is useful for visually excluding an outlier without deleting its numerical analysis.

---

# 4. Analysis tab

## 4.1 Detrending method

### Polynomial regression

Fits a polynomial growth trend and subtracts it from the measured biomass.

### Logistic regression

Fits a logistic growth curve and subtracts it from the measured biomass. The fit-search interval and signal-analysis interval can be different.

---

# 5. Common analysis settings

## Smoothing kernel [points]

Number of measurements used for the moving-average smoothing of the residual signal.

A larger kernel produces stronger smoothing and removes more short-term noise, but can reduce or distort faster oscillations. A smaller kernel preserves more detail but also more noise.

This setting is expressed in **data points**, not hours. Therefore, the same kernel corresponds to a different time span when the sampling interval changes.

For example, with a 20 s measurement interval:

```text
1200 points × 20 s = 24000 s = 6.67 h
```

## FFT zero padding multiplier

Adds zeros after the analyzed smoothed signal before the FFT.

Zero padding makes the displayed FFT frequency grid denser and can make the dominant peak easier to inspect. It does **not** create additional biological information or increase the true frequency resolution of the experiment.

Default: `4`.

## Manual measurement interval [s]

Sampling interval used for FFT frequency conversion when automatic interval detection is disabled.

Example: if one measurement is taken every 20 seconds, enter `20`.

## Auto-detect FFT measurement interval from Relative Time

Recommended in most cases.

When enabled, the app calculates the median spacing between consecutive Relative Time values and uses this interval for FFT calculations. This keeps frequencies and periods correct when datasets were measured at different sampling rates.

When disabled, the **Manual measurement interval [s]** value is used instead.

---

# 6. Polynomial settings

## Analysis start [h]

First timepoint included in the polynomial detrending analysis.

## Analysis end [h]

Last timepoint included in the polynomial detrending analysis.

Time boundaries are inclusive when a measurement exists exactly at the boundary.

## Polynomial degree

Controls the flexibility of the fitted polynomial biomass trend.

- Lower degree: smoother, less flexible trend.
- Higher degree: more flexible trend and greater risk of following the oscillation itself.

Default: `4`.

The degree is selected manually. Regression R² is reported as a fit-quality value but does not automatically choose the degree.

## Polynomial workflow

1. Select the requested time interval.
2. Fit polynomial time terms of the chosen degree.
3. Predict the slowly changing biomass trend.
4. Calculate residual = measured biomass − polynomial fit.
5. Subtract the arithmetic mean of the residual so it is centered around zero.
6. Smooth the residual using the selected moving-average kernel.
7. Use the smoothed residual for FFT and cosine analysis.

---

# 7. Logistic settings

The logistic growth model is fitted independently from the downstream signal interval.

## Fit start search min [h]

Earliest candidate starting time for the logistic fit.

## Fit start search max [h]

Latest candidate starting time for the logistic fit.

If minimum and maximum are identical, only one starting time is tested.

## Fit end search min [h]

Earliest candidate ending time for the logistic fit.

## Fit end search max [h]

Latest candidate ending time for the logistic fit.

If minimum and maximum are identical, only one ending time is tested.

## Grid points per fit boundary

Number of candidate values generated between each minimum and maximum search boundary.

More grid points test more possible fitting windows and can find a better R², but require more fitting operations.

If a boundary minimum and maximum are identical, only that fixed boundary is used regardless of the grid-point value.

## Signal analysis start [h]

First timepoint where the selected logistic curve is evaluated and subtracted from the measured biomass for the **real analytical signal**.

This can be different from the logistic fit start.

## Signal analysis end [h]

Last timepoint used for the real analytical residual, FFT, cosine fit, period/amplitude/phase calculation, and extrema detection.

This can extend beyond the logistic fitting window, but doing so is an extrapolation of the growth model. Always inspect the biomass + logistic-fit plot to confirm that the fitted curve is still biologically reasonable in that region.

## Gaussian sigma

Used during adaptive lag-phase detection in the logistic fitting procedure.

The biomass trace is Gaussian-smoothed before its derivative is calculated. A larger sigma suppresses short-term fluctuations more strongly before the lag/growth transition is estimated.

This Gaussian smoothing is part of logistic fit initialization and is separate from the final residual moving-average smoothing kernel.

Default: `2`.

## Slope fraction

Fraction of the maximum smoothed biomass slope used as the threshold for detecting the onset of substantial growth.

Conceptually:

```text
threshold = slope_fraction × maximum derivative
```

The first point whose derivative exceeds this threshold is used as an estimate of the transition into active growth and helps initialize the logistic model.

Default: `0.05` = 5% of the maximum detected slope.

A larger value detects a later/stronger rise. A smaller value can detect an earlier weaker rise and may be more sensitive to noise.

## Signal normalization

### Mean-centered residual [a.u.]

Calculates:

```text
residual = measured biomass − fitted biomass
signal = residual − arithmetic mean(residual)
```

The result remains in backscatter arbitrary units.

### Residual / fitted biomass [relative]

Calculates:

```text
signal = (measured biomass − fitted biomass) / fitted biomass
```

The result is dimensionless and represents the residual as a fraction of the fitted biomass.

Example: `0.05` means the measured value is 5% above the fitted biomass trend at that timepoint.

Points where fitted biomass is effectively zero are left undefined to avoid unstable division.

Changing the normalization does not change the logistic fit itself or its R². It changes the signal subsequently passed to smoothing, FFT, and cosine analysis.

---

# 8. Optional display-only logistic fit

## Add separate display-only logistic fit for omitted timepoints

Creates a second logistic fit used only to visually extend the signal before the real analytical signal begins.

It does **not** contribute to:

- FFT
- cosine fitting
- period
- amplitude
- phase
- peak/trough detection
- primary R² selection
- group statistics
- biological summary values

Failure of the display-only fit does not invalidate the main analysis.

## Visual-only fit start [h]

Start of the independent visual-only logistic fitting interval.

## Visual-only fit end [h]

End of the independent visual-only logistic fitting interval.

The fit interval may extend beyond the time actually displayed. The visual-only residual is automatically hidden once the real analytical signal starts.

## Dashed divider: visual-only → analytical signal

Plotting checkbox found under **Plotting → Signal / FFT layers**.

When enabled, a dashed vertical line is drawn at the transition from the display-only signal to the real analytical signal.

This divider is purely graphical and does not change any calculations.

---

# 9. FFT and cosine analysis

The smoothed analytical residual is passed to the FFT.

The FFT is used to identify the dominant oscillatory frequency and corresponding period. The resulting values are also used to initialize the cosine fit.

The fitted model is:

```text
y(t) = A × cos(2π f t + φ)
```

where:

- `A` = cosine amplitude
- `f` = fitted frequency
- `1/f` = cosine period
- `φ` = fitted phase angle

## Cosine period [h]

Period of the fitted cosine in hours.

## Cosine amplitude

Amplitude of the fitted cosine in the units of the selected signal normalization.

## Cosine phase shift [h]

Signed fitted phase parameter converted from radians into hours of the fitted cycle.

## Cosine peak phase [h]

More intuitive phase output describing where a cosine maximum occurs within one fitted period, measured relative to **Relative Time = 0**.

It is calculated as:

```text
peak phase = (−phase shift) modulo fitted period
```

For biological comparisons, time zero should represent the same biological reference event in all samples.

If nonlinear cosine fitting fails, the app can fall back to the FFT-based cosine approximation. The summary indicates when this fallback was used.

---

# 10. Plotting tab — general appearance

## Output format

Available values:

- `pdf` — vector PDF
- `svg` — vector SVG
- `png` — raster PNG
- `both` — PDF + SVG
- `all` — PDF + SVG + PNG

PDF and SVG are vector formats and remain sharp when resized. PNG resolution depends on the DPI setting.

## Figure width [in]

Width of exported figures in inches.

## Figure height [in]

Height of exported figures in inches.

## DPI

Raster resolution used mainly for PNG export. Default: `600` DPI.

## Data line width

Thickness of raw/processed data lines.

## Data line style

Line style used for normal data traces:

- `-` solid
- `--` dashed
- `-.` dash-dot
- `:` dotted

## Fit line width

Thickness of regression/cosine fit lines.

## Fit line style

Line style used for regression fits and ordinary cosine fit display.

## Axis font size

Font size used for axis labels and most tick-related text.

## Title font size

Font size used for figure titles.

## Legend font size

Font size used in plot legends.

## Regression fit color

Color used for individual regression growth-trend lines.

## Cosine fit color

Color used for the cosine line in individual signal plots.

In multi-group summary comparison plots, group colors are used where necessary to keep multiple curves distinguishable.

---

# 11. Plotting tab — axes and plot ranges

## Time X start [h]

Optional displayed lower limit of time-axis plots. Leave blank for automatic/full range.

## Time X end [h]

Optional displayed upper limit of time-axis plots. Leave blank for automatic/full range.

These values affect the visible axis only; they do not change the analysis interval.

## Cosine plot start [h]

Optional first time where the already-fitted cosine line is drawn.

Leave blank to use the complete available cosine curve.

This is a **plot-only** setting. It does not refit the cosine and does not change period, amplitude, phase, peak phase, FFT, or summary values.

## Cosine plot end [h]

Optional final time where the already-fitted cosine line is drawn.

Leave blank to draw it to the end of its available range.

This is also plot-only.

## Biomass Y min / Biomass Y max

Optional manual Y-axis limits for biomass plots.

## Signal Y min / Signal Y max

Optional manual Y-axis limits for residual/signal plots.

## FFT X min [1/h] / FFT X max [1/h]

Displayed frequency range for FFT figures.

Default range: `0` to `0.3 1/h`.

## FFT Y min / FFT Y max

Optional manual Y-axis limits for FFT amplitude.

## Major tick interval [h]

Spacing of major time-axis ticks when major/minor time marks are enabled.

Default: `24 h`.

## Minor tick interval [h]

Spacing of minor time-axis ticks.

Default: `12 h`.

## Use major/minor time marks

Turns configured major and minor tick intervals on/off.

## Show grid

Shows or hides plot grid lines.

---

# 12. Plotting tab — signal / FFT layers

## Raw signal

Shows the unsmoothed residual signal.

## Smoothed signal

Shows the moving-average-smoothed residual used for FFT/cosine analysis.

## Cosine fit

Shows the fitted cosine line. The optional cosine plot start/end values can restrict only where this line is displayed.

## Dashed divider: visual-only → analytical signal

Shows the visual-only/main-analysis boundary when the optional second logistic fit is active.

## Peak/trough markers

Marks detected extrema in the smoothed analytical signal.

## Annotate extrema times

Writes the corresponding time value next to detected extrema.

## Annotate FFT period

Adds the dominant FFT-period annotation where supported by the plot type.

## Peak width [points]

Minimum peak width used by the extrema detector, expressed in measurement points.

Larger values require broader peaks/troughs and can suppress narrow/noisy extrema.

## Peak distance [points]

Minimum separation between detected extrema, expressed in data points.

Larger values prevent closely spaced fluctuations from being counted as separate extrema.

## Biomass alpha

Transparency of biomass lines.

`1.0` = fully opaque, lower values = more transparent.

## Raw signal alpha

Transparency of raw residual lines.

## Smoothed signal alpha

Transparency of smoothed residual lines.

## Regression fit alpha

Transparency of regression growth-fit lines.

## Cosine fit alpha

Transparency of fitted cosine lines.

---

# 13. Light / dark entrainment display

These options add visual time annotations only and do not affect calculations.

## Annotate entrainment until [h]

Time until which light/dark entrainment cycles should be shown.

`0` disables entrainment annotation.

## Light duration [h]

Duration of each light phase.

Default: `12 h`.

## Dark duration [h]

Duration of each dark phase.

Default: `12 h`.

---

# 14. Legends and overview layout

## Save separate group legend

Exports a standalone legend figure for reuse when assembling multi-panel figures.

## Place plot legends below axes

Places legends beneath the plotting area instead of inside the axes.

## Legend columns

Number of columns used for legends.

## Overview columns

Number of subplot columns used in overview-grid figures.

---

# 15. Which groups / samples contribute to plots

The group selector controls the normal/global plot sets.

Select one or multiple groups with Cmd-click on macOS or Ctrl-click on Windows/Linux.

A sample must satisfy both conditions to appear in normal plot sets:

1. its **Plot** checkbox is enabled in Samples & groups;
2. its group is selected in the main group selector.

## Select all groups

Selects every currently available group for normal plots.

## Deselect all groups

Removes all groups from normal plot generation. Analysis and data export still proceed.

## Refresh groups from Samples tab

Updates the group selector after editing group names in Samples & groups.

---

# 16. Custom group comparisons

Custom comparisons are independent saved combinations of groups.

To create one:

1. select two or more groups in the group selector;
2. click **Add selected groups as comparison**;
3. optionally change the main selector and add another combination.

Examples:

```text
WT + ΔkaiC
WT + ΔkaiABC
WT + Treatment A + Treatment B
```

A saved comparison remembers its own groups. Changing or clearing the **main group selector later does not remove those groups from the saved comparison**.

The sample-level **Plot** checkbox still applies. Therefore, an unchecked replicate/outlier remains excluded from comparisons.

Each saved comparison produces:

### Replicate-level plots

- replicate biomass + regression fits
- replicate smoothed signals
- replicate FFT spectra

### Group mean plots

- mean biomass + mean fit, with biomass ± SD shading
- mean smoothed signal, with signal ± SD shading
- FFT calculated from the mean smoothed signal
- FFT period view calculated from the mean smoothed signal

### Group median plots

- median biomass + median fit
- median smoothed signal
- FFT calculated from the median smoothed signal
- FFT period view calculated from the median smoothed signal

### Additional descriptive comparison plots

- cosine period by group
- cosine amplitude by group
- cosine peak phase by group
- regression R² by group

Replicates are displayed as individual points. A diamond marks the arithmetic mean and a horizontal marker shows the median.

A `comparison_sample_metrics.csv` file is also exported for each comparison, together with per-group statistics CSV files.

---

# 17. Plot sets to generate

## Individual plots

Creates biomass-fit, signal, FFT, and optional FFT-period figures for each included sample.

## Overview grids

Creates multi-panel pages containing all selected samples and condition-specific replicate panels.

## Replicate overlays

Overlays replicates belonging to the same group.

## Condition mean ± SD

Creates group-level mean plots and median plots from the included replicates.

## All conditions together

Creates figures containing all selected groups together.

## Group by replicate number

Attempts to identify replicate numbers from sample names and plots matching replicate numbers across conditions.

## Additional FFT period view

Adds an alternative FFT view with period rather than frequency on the X axis where supported.

---

# 18. Live preview

The Live preview tab recalculates the selected sample using the current analysis settings and displays the selected plot type.

It is intended for checking fit quality, smoothing, axis settings, visual-only behavior, and cosine display before processing the full experiment.

Preview changes do not modify the source data.

---

# 19. Output files

The main output directory contains files such as:

```text
backscatter_analysis_app/
    biological_summary.xlsx
    biological_summary_by_group.csv
    biological_summary_by_group_all_samples.csv
    biological_summary_by_sample.csv
    analysis_summary.csv
    settings.json
    regression_data_all_samples.csv
    regression_data_all_samples.xlsx
    analyzed_excel_exports/
    groups/
    plots/
    samples/
```

## biological_summary.xlsx

Contains presentation-ready and machine-readable biological summary sheets.

Typical values include:

- cosine period
- cosine amplitude
- cosine peak phase
- FFT period
- regression R²
- signal normalization
- cosine FFT-fallback count

## analysis_summary.csv

One row per successfully analyzed sample with detailed numerical results and fitting information.

## settings.json

Stores the complete analysis, plotting, group, label, color, and comparison settings used for the run.

## regression_data_all_samples.csv / .xlsx

Wide table containing time, biomass, fitted trend, residual signal, smoothed signal, cosine curve, and optional visual-only traces for all analyzed samples.

## groups/

Stores condition/group statistics independently of the plotted comparison figures.

## plots/08_custom_group_comparisons/

Contains one folder per saved comparison with replicate, mean, median, FFT, period-view, metric, and comparison-statistics outputs.

---

# 20. Per-sample outputs

Each sample receives detailed CSV and Excel outputs including:

- time
- measured biomass
- regression fit
- raw residual signal
- smoothed analytical signal
- fitted cosine
- optional visual-only logistic fit
- optional visual-only signal
- optional visual-only smoothed signal
- FFT spectrum
- numerical summary values

The visual-only columns remain separate from the real analytical signal columns so they cannot be confused with values used for FFT/cosine calculations.

---

# 21. Interpreting mean and median group plots

## Mean

The arithmetic mean is the sum of replicate values divided by the number of replicates at that timepoint.

Mean plots are useful for representing the average response, but can be more influenced by an extreme replicate.

## Median

The median is the middle replicate value after sorting the replicate values at that timepoint.

Median plots can be useful when one replicate differs strongly from the others.

The comparison plots provide both rather than replacing one with the other.

---

# 22. Important interpretation notes

## Regression R²

R² describes how well the selected growth model reproduces the biomass values in its fitting interval. A high R² does not by itself prove that the model is biologically appropriate. Always inspect the fit visually.

## Logistic extrapolation

If the signal-analysis interval extends beyond the logistic fitting interval, the logistic curve is being extrapolated. Use this only if the curve continues to represent the biomass trend sensibly.

## Smoothing

Very strong smoothing can reduce oscillation amplitude or remove real structure. Very weak smoothing can leave high-frequency noise that complicates extrema and FFT interpretation.

## Zero padding

Zero padding improves the appearance/interpolation of the FFT spectrum but does not add experimental measurements.

## Phase comparison

Cosine peak phase should only be compared biologically if Relative Time = 0 has the same meaning in every compared sample.

## Plot-only controls

The following settings change only figure appearance and do not recalculate the biological analysis:

- Time X start/end
- Cosine plot start/end
- Y-axis limits
- FFT display limits
- line widths/styles/colors/alpha
- grid and tick display
- visual-only dashed divider
- light/dark annotation
- legend layout

---

# 23. macOS

Run:

```text
run_backscatter_app.command
```

If macOS Gatekeeper blocks it, right-click the file and choose **Open** once.

To build a local `.app`:

```text
build_mac_app.command
```

The generated application is placed under:

```text
dist/Backscatter Analysis.app
```

---

# 24. Windows

Install a standard Python 3 distribution from python.org with Tcl/Tk support.

Run:

```text
run_backscatter_app_windows.bat
```

To build a local Windows application:

```text
build_windows.bat
```

The generated application is placed under:

```text
dist\Backscatter Analysis\
```

---

# 25. Linux

Python 3 and Tkinter are required. Tkinter may need to be installed through the system package manager.

Examples:

```text
Ubuntu/Debian: sudo apt install python3-tk
Fedora:        sudo dnf install python3-tkinter
Arch:          sudo pacman -S tk
```

Run:

```text
chmod +x run_backscatter_app_linux.sh
./run_backscatter_app_linux.sh
```

Build:

```text
chmod +x build_linux.sh
./build_linux.sh
```

---

# 26. Python dependencies

See `requirements.txt`.

Main packages:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- openpyxl

Tkinter is normally included with standard macOS/Windows Python installations and is often packaged separately on Linux.

---

# 27. Cross-platform note

The Python application is cross-platform, but PyInstaller applications are platform-specific. Build the macOS app on macOS, the Windows executable on Windows, and the Linux executable on Linux.
