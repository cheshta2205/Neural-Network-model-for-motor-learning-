# PlottingTools

Lightweight plotting utilities used in this project for consistent, publication-style Matplotlib figures.

This package currently provides helper functions for:

- global Matplotlib style and rcParams setup
- axis sharing and cleanup
- adding fitted lines and confidence-style filled regions
- adding scale bars and cleaner axes/spines
- plotting 2D trajectory means with shaded envelopes

## Package Layout

- `plotting_functions.py`: all plotting helper functions
- `__init__.py`: package marker (currently empty)
- `setup.py`: minimal setuptools config (`name='pretty_plots'`)
- `requirements.txt`: pinned Python dependencies
- `environment.yml`: Conda environment specification

## Installation

From this directory (`src/PlottingTools`), you can install in editable mode:

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

With Conda:

```bash
conda env create -f environment.yml
conda activate habit
```

## Quick Start

```python
import matplotlib.pyplot as plt
import numpy as np

from plotting_functions import set_plt_params, abline, shaded_regions_2d

set_plt_params(font_size=8, style="ggplot", figsize=(5, 4))

x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(scale=1.0, size=x.shape)

fig, ax = plt.subplots()
ax.scatter(x, y, s=8, alpha=0.6)
abline(slope=2.0, intercept=1.0, axes=ax, color="tab:red", label="fit")
ax.legend()

plt.show()
```

## Function Reference

- `identify_axes(ax_dict, fontsize=48)`:
	writes large centered labels on each axis for debugging subplot layouts.

- `set_share_axes(axs, target=None, sharex=False, sharey=False)`:
	manually links x/y axes across a list or NumPy array of axes and hides redundant tick labels.

- `abline(slope, intercept, color='red', label=None, axes=None, ls='--', exponential=False, **kwargs)`:
	draws a line from slope/intercept using current axis limits.

- `abline_fill_between(slope, intercept, slope2=None, intercept2=None, color='red', axes=None, exponential=False, **kwargs)`:
	fills the area between two lines (or around one line if only one is provided).

- `set_plt_params(...)`:
	applies a consistent rcParams style preset for font sizes, line widths, ticks, figure format, and spine visibility.

- `arrow_plot(axes, x, y, z=None, narrs=30, dspace=0.5, direc='pos', color='black')`:
	computes trajectory segment distances (note: function is currently incomplete and does not draw arrows).

- `make_axis_nice(ax=None, offset=10, line_width=0.5, spines=None, color=None)`:
	offsets and styles selected spines for cleaner axis presentation.

- `add_scale_bars(ax=None, x_bar=0, y_bar=None, bar_width=2, hide_axis=True, x_direction='left', y_offset=0, x_offset=0, preserve_axis=True, x_text_scale=5, y_text_scale=5)`:
	adds x/y scale bars to one axis or multiple axes.

- `round_lims(ax=None, round_x=False, round_y=False)`:
	rounds axis locators to integer tick values and removes margins.

- `shaded_regions_2d(x_mean, y_mean, x_std, y_std, ax=None, alpha=0.2, ls='--', **kwargs)`:
	plots a mean 2D trajectory and shaded region based on local normal direction.

## Notes

- The package metadata name in `setup.py` is currently `habit_project`, which does not match this folder name.
- Most functions rely on NumPy arrays and Matplotlib `Axes` objects.
- `set_plt_params()` resets Matplotlib defaults before applying the selected style.
