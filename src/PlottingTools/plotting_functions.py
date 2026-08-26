"""This script contains common function for improving the quality of plots

"""

from typing import Tuple, Union, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def set_share_axes(axs, target=None, sharex=False, sharey=False):

    # Manage share using grouper objects
    if isinstance(axs, np.ndarray):
        if target is None:
            target = axs.flatten()[0]
        for ax in axs.flatten():
            if sharex:
                target._shared_axes['x'].join(target, ax)
            if sharey:
                target._shared_axes['y'].join(target, ax)
    else:
        target = axs[0]
        for ax in axs:
            if sharex:
                target._shared_axes['x'].join(target, ax)
            if sharey:
                target._shared_axes['y'].join(target, ax)

    if isinstance(axs, np.ndarray):
        # Turn off x tick labels and offset text for all but the bottom row
        if sharex and axs.ndim > 1:
            for ax in axs[:-1, :].flatten():
                ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
                ax.xaxis.offsetText.set_visible(False)
        # Turn off y tick labels and offset text for all but the left most column
        if sharey and axs.ndim > 1:
            for ax in axs[:, 1:].flatten():
                ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
                ax.yaxis.offsetText.set_visible(False)
        elif sharey and axs.ndim == 1:
            for ax in axs[1:].flatten():
                ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
                ax.yaxis.offsetText.set_visible(False)


def abline(slope, intercept, color: str = 'red', label: str = None, axes=None,
           ls: str = '--', exponential: bool = False, **kwargs):
    """Plot a line from slope and intercept"""
    if axes is None:
        axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    if not exponential:
        y_vals = intercept + slope * x_vals
    else:
        y_vals = np.exp(intercept + slope * x_vals)

    if label is None:
        axes.plot(x_vals, y_vals, ls=ls, color=color, **kwargs)
    else:
        axes.plot(x_vals, y_vals, ls=ls, label=label, color=color, **kwargs)


def abline_fill_between(slope, intercept, slope2: float = None, intercept2: float = None,
                        color: str = 'red', axes=None, exponential: bool = False, **kwargs):
    """Plot a line from slope and intercept"""

    if slope2 is None:
        slope2 = slope

    if intercept2 is None:
        intercept2 = intercept

    if axes is None:
        axes = plt.gca()

    x_vals = np.array(axes.get_xlim())
    if not exponential:
        y_vals = intercept + slope * x_vals
        y_vals2 = intercept2 + slope2 * x_vals
    else:
        y_vals = np.exp(intercept + slope * x_vals)
        y_vals2 = np.exp(intercept2 + slope2 * x_vals)

    axes.fill_between(x_vals, y_vals, y_vals2, color=color, **kwargs)


def set_plt_params(font_size: int = 6, legend_size: int = 6, axes_label_size: int = 6, linewidth: float = 1.5,
                   axes_title_size: int = 6, xtick_label_size: int = 5, ytick_label_size: int = 5,
                   ticksize: float = 2, path_simplify: bool = False,
                   fig_title_size: int = 7, style: str = 'ggplot', font: str = 'avenir',
                   file_format: str = 'svg', fig_backend: str = 'svg',
                   fig_dpi: int = 500, figsize: Tuple[float, float] = (11, 8),
                   auto_method: str = None, x_margin: float = None, y_margin: float = None,
                   render_path: bool = False, marker_size: float = 2):
    """
    This function sets the plot parameters for a plot
    :param font_size:
    :param legend_size:
    :param axes_label_size:
    :param linewidth:
    :param axes_title_size:
    :param xtick_label_size:
    :param ytick_label_size:
    :param ticksize:
    :param path_simplify:
    :param fig_title_size:
    :param style:
    :param font:
    :param file_format:
    :param fig_dpi:
    :param figsize:
    :param auto_method:
    :param x_margin:
    :param y_margin:
    :return:
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(style)
    plt.rcParams['savefig.format'] = file_format
    plt.rcParams['backend'] = fig_backend
    plt.rcParams['savefig.dpi'] = fig_dpi
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['path.simplify'] = path_simplify
    plt.rcParams['lines.markersize'] = marker_size
    if not auto_method is None:
        plt.rcParams['axes.autolimit_mode'] = auto_method
    if not x_margin is None:
        assert ((x_margin >= 0) & (x_margin <= 1))
        plt.rcParams['axes.xmargin'] = x_margin
    if not y_margin is None:
        assert ((y_margin >= 0) & (y_margin <= 1))
        plt.rcParams['axes.ymargin'] = y_margin

    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=axes_title_size)
    plt.rc('axes', labelsize=axes_label_size)
    plt.rc('xtick', labelsize=xtick_label_size)
    plt.rc('ytick', labelsize=ytick_label_size)
    plt.rc('legend', fontsize=legend_size)
    plt.rc('figure', titlesize=fig_title_size)
    plt.rc('lines', linewidth=linewidth, markersize=marker_size)
    plt.rcParams['font.family'] = font
    plt.rcParams['xtick.major.size'] = ticksize
    plt.rcParams['ytick.major.size'] = ticksize
    plt.rcParams['xtick.minor.size'] = ticksize / 3
    plt.rcParams['ytick.minor.size'] = ticksize / 3

    if not render_path:
        plt.rcParams['svg.fonttype'] = 'none'

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True


def arrow_plot(axes, x, y, z=None, narrs=30, dspace=0.5,
               direc='pos', color='black'):

    if z is not None:
        three_dimensional_plot = True
    else:
        three_dimensional_plot = False

    # Get distances between points
    r = [0]
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        if three_dimensional_plot:
            dz = z[i] - z[i - 1]
            r.append(np.sqrt(dx ** 2 + dy ** 2 + dz **2))
        else:
            r.append(np.sqrt(dx ** 2 + dy ** 2))


def make_axis_nice(ax: Union[plt.Axes, plt.Figure] = None, offset: int = 10,
                   line_width: float = 0.5, spines: List = None, color: str = None):
    """ Makes axis pretty
    This function modifies the x and y axis so that there is a vertical and horizontal gap between them
    Args:
        ax: This is the axis (axes) that need to be changed. When this argument is a fig all axes of the fig
        get modified
        offset: Size of the gap.
        line_width: Linew width of the new axes.
        spines:
        color:
    """
    if ax is None:
        ax = plt.gca()

    if spines is None:
        spines = ['left', 'bottom']
    if type(ax) == plt.Figure:
        ax_list = ax.axes
    else:
        ax_list = [ax]

    for ax in ax_list:
        for spine in spines:
            ax.spines[spine].set_linewidth(line_width)
            if color is not None:
                ax.spines[spine].set_color(color)
            ax.spines[spine].set_position(('outward', offset))


def add_scale_bars(ax: Union[plt.Axes, plt.Figure] = None, x_bar: Optional[float] = 0,
                   y_bar: Optional[float] = None, bar_width: float = 2, hide_axis: bool = True,
                   x_direction: str = 'left', y_offset: float = 0, x_offset: float = 0,
                   preserve_axis: bool = True, x_text_scale: float = 5,
                   y_text_scale: float = 5):
    """
    Add a scale bar to the given axis/axes
    Args:
        ax: Axes to add the scale bar to
        x_bar: Size of the x_bar
        y_bar: Size of the y_bar
        bar_width: Bar line width, by default 2
        hide_axis: Whether to hide the original axis ticks, by default True
        x_direction: Whether to start x_bar from the right or left side, by default left
        y_offset: Offset used for x_bar in the y dimension
        x_offset: Offset used for the y_bar in the x dimension
        preserve_axis: Whether to use the same lims across all the axes, by default True
        x_text_scale: Scale for offsetting the x_label for the x_bar
        y_text_scale: Scale for offsetting the y_label for the y_bar

    Returns:

    """
    if ax is None:
        ax = plt.gca()

    if type(ax) == plt.Figure:
        ax_list = ax.axes
    elif isinstance(ax, list):
        ax_list = ax
    else:
        ax_list = [ax]

    x_lims, y_lims = None, None
    for axidx, axis in enumerate(ax_list):
        if y_lims is None:
            y_lims = axis.get_ylim()
        if x_lims is None:
            x_lims = axis.get_xlim()
        print(y_lims, x_lims)
        if x_bar is not None:
            if hide_axis:
                axis.xaxis.set_visible(False)

            y_range = np.ptp(y_lims)
            if x_direction == 'left':
                axis.hlines(y_lims[0] - y_offset * y_range,
                            xmin=x_lims[0], xmax=x_lims[0] + x_bar, lw=bar_width,
                            color='k')
                if axidx == 0:
                    axis.text(x_lims[0], y_lims[0] - x_text_scale * y_offset * y_range, f"{x_bar}",
                          horizontalalignment='center', verticalalignment='center')

            elif x_direction == 'right':
                axis.hlines(y_lims[0] - y_offset,
                            xmin=x_lims[1] - x_bar, xmax=x_lims[1], lw=bar_width,
                            color='k')
                if axidx == 0:
                    axis.text(x_lims[1] - x_bar, y_lims[0] - x_text_scale * y_offset, f"{x_bar}",
                              horizontalalignment='center', verticalalignment='center')

        if y_bar is not None:
            axis.yaxis.set_visible(False)
            x_range = np.ptp(x_lims)

            axis.vlines(x_lims[0] - x_offset * x_range,
            ymin=y_lims[0], ymax=y_lims[0] + y_bar, color='k')
            if axidx == 0:
                axis.text(x_lims[0] - y_text_scale * x_offset * x_range, y_lims[0], f"{y_bar}",
                      horizontalalignment='center', verticalalignment='center',
                      rotation='vertical')

        # remove spines
        if hide_axis:
            if x_bar is not None:
                axis.xaxis.set_visible(False)
                axis.spines['bottom'].set_visible(False)
            if y_bar is not None:
                axis.yaxis.set_visible(False)
                axis.spines['left'].set_visible(False)

        if not preserve_axis:
            x_lims, y_lims = None, None


def round_lims(ax: Union[plt.Axes, plt.Figure] = None, round_x: bool = False,
               round_y: bool = False):
    """
    Rounds axes to round numbers of a given precision and number of ticks.
    Args:
        ax: Axes which will be rounded. Can be a matplotlib Axes or a Figure
        round_x: Whether to round the x axis, by default False
        round_y: Whether to round the y axis, by default False

    Returns:

    """

    if ax is None:
        ax = plt.gca()

    if type(ax) is plt.Figure:
        ax_list = ax.axes
    elif isinstance(ax, list):
        ax_list = ax
    else:
        ax_list = [ax]

    for axis in ax_list:

        if round_x:
            axis.locator_params(axis='x', integer=True, tight=True)

        if round_y:
            axis.locator_params(axis='y', integer=True, tight=True)
        axis.margins(0, 0)


def shaded_regions_2d(x_mean, y_mean, x_std, y_std, ax: plt.Axes = None,
                      alpha: float = 0.2, ls: str = '--', **kwargs):

    if ax is None:
        ax = plt.gca()

    # Compute trajectory tangent and normal directions
    dx = np.gradient(x_mean)
    dy = np.gradient(y_mean)
    norm = np.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm

    # Normal vector (perpendicular)
    nx = -dy
    ny = dx

    # Use average of x_std and y_std for thickness, or customize
    thickness = 0.5 * (x_std + y_std)

    # Upper and lower boundary points
    x_upper = x_mean + nx * thickness
    y_upper = y_mean + ny * thickness
    x_lower = x_mean - nx * thickness
    y_lower = y_mean - ny * thickness

    # Construct the shaded region polygon
    x_poly = np.concatenate([x_upper, x_lower[::-1]])
    y_poly = np.concatenate([y_upper, y_lower[::-1]])

    # Plot
    ax.plot(x_mean, y_mean, ls=ls, **kwargs)
    ax.fill(x_poly, y_poly, alpha=alpha, **kwargs)


