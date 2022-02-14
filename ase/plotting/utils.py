"""Contains base plots and matplotlib display settings."""
import pickle
from pathlib import Path
from functools import partial
import itertools
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import seaborn as sns
from scipy.stats import ks_2samp, wilcoxon, mannwhitneyu
from seaborn.rcmod import plotting_context

from ase.visualize import Visualiser
""""🔥🔥🔥 Set some global params and plotting settings."""

# Font options
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}\usepackage{amssymb}')

# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
fs = 9
label_fs = fs # - 1
family = 'serif'
rcParams['font.family'] = 'serif'
# rcParams['font.sans-serif'] = ['Times']
rcParams['font.sans-serif'] = [
    'DejaVu Sans',
    'Bitstream Vera Sans',
    'Computer Modern Sans Serif']

rcParams['font.size'] = fs

prop = dict(size=fs)
legend_kwargs = dict(frameon=True, prop=prop)
new_kwargs = dict(prop=dict(size=fs-2))

# Styling
c = 'black'
rcParams.update({'axes.edgecolor': c, 'xtick.color': c, 'ytick.color': c})
# ICLR
# textwidth = 5.50107

# AISTATS
textwidth = 6.75133
linewidth = 3.2506


# Global Names (Sadly not always used)
# acquisition_step_label = 'Acquired Points'
ACQUISITION_STEP_LABEL = r'N\textsuperscript{\underline{o}} Acquired Points'
LABEL_ACQUIRED_DOUBLE = 'Acquired Points'
LABEL_ACQUIRED_FULL = 'Number of Acquired Test Points'
# diff_to_empircal_label = r'Difference to Full Test Loss'
LABEL_DIFF_TRUE = 'Difference to \n True Expectation'
diff_to_empircal_label = r'Difference to True Test Risk'
DIFF_EMPIRICAL_TWOLINES = 'Difference to \n True Test Risk'
std_diff_to_empirical_label = 'Standard Deviation of Estimator Error'
sample_efficiency_label = 'Efficiency'
# LABEL_RANDOM = r'Naïve Monte Carlo'
# LABEL_RANDOM = r'Naïve MC'
LABEL_RANDOM = r'MC'
POOL_LIMIT_LABEL = 'Pool Limit'
LABEL_STD = 'Squared Error'
# LABEL_STD = 'Estimator MSE'
# LABEL_ERROR =
LABEL_MEDIAN = 'Median \n Squared Error'
LABEL_RELATIVE_COST = 'Relative Labeling Cost'
LABEL_MEAN_LOG = 'Mean Log Squared Error'
# IS_LABEL = 'Importance Sampling'
IS_LABEL = 'AIS'
ASMC_LABEL = 'ASE'
LURE_LABEL = 'LURE'
LABELS_DEFAULT = [LABEL_RANDOM, IS_LABEL, ASMC_LABEL]
LABELS_DEFAULT_AT = [LABEL_RANDOM, LURE_LABEL, ASMC_LABEL]


# Color palette
# CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                #   '#f781bf', '#a65628', '#984ea3',
                #   '#999999', '#e41a1c', '#dede00']
# CB_color_cycle = [CB_color_cycle[i] for i in [0, 1, 2, -2, 5, 4, 3, -3, -1]]
# cbpal = sns.palettes.color_palette(palette=CB_color_cycle)
pal = sns.color_palette('colorblind')
# pal[5], pal[6], pal[-2] = cbpal[5], cbpal[6], cbpal[-1]
# pal[2] = [i/255 for i in [0, 212, 152]]
pal[-1], pal[-2] = pal[-2], pal[-1]


RANDOM_COLOR = pal[4]
IS_COLOR = pal[-1]
ASMC_COLOR = pal[2]

ACQRISK2COLOR = dict(
    RandomAcquisition_BiasedRiskEstimator=RANDOM_COLOR,
    AnySurrogateAcquisitionEntropy_LazySurrEnsemble_FancyUnbiasedRiskEstimator=IS_COLOR,
    AnySurrogateAcquisitionEntropy_LazySurrEnsembleLarge_FancyUnbiasedRiskEstimator=IS_COLOR,
    SelfSurrogateAcquisitionEntropy_LazySurr_FancyUnbiasedRiskEstimator=IS_COLOR,
    SelfSurrogateAcquisitionEntropy_Surr_FancyUnbiasedRiskEstimator=IS_COLOR,
    AnySurrogateAcquisitionEntropy_LazySurrEnsembleNoSample_QuadratureRiskEstimator=ASMC_COLOR,
    AnySurrogateAcquisitionEntropy_LazySurrEnsembleLargeNoSample_QuadratureRiskEstimator=ASMC_COLOR,
    SelfSurrogateAcquisitionEntropy_LazySurrNoSample_QuadratureRiskEstimator=ASMC_COLOR,
    SelfSurrogateAcquisitionEntropy_LazySurr_QuadratureRiskEstimator=ASMC_COLOR,
    SelfSurrogateAcquisitionEntropy_Surr_QuadratureRiskEstimator=ASMC_COLOR,
    )


""""🔥🔥🔥 General purpose classes and functions."""


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
#     def _set_format(self, vmin=None, vmax=None):
#         self.format = self.fformat
#         if self._useMathText:
#             self.format = r'$\mathdefault{%s}$' % self.format


def resolve_color(acqrisk, acqrisk2color):
    if (acqrisk2color is not None) and (c:=acqrisk2color.get(acqrisk, False)):
        return c
    elif c := ACQRISK2COLOR.get(acqrisk, False):
        return c
    else:
        print(f'Color for {acqrisk} could not be found anywhere, choosing random')
        return np.random.rand(3)


def print_palette_latex():
    """Print color palette in format compatible with latex."""
    for i, p in enumerate(pal):
        bs = '\\'
        print(
            f'{bs}definecolor{{pal{i}}}{{rgb}}{{{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}}}')


def get_visualisers(paths, **kwargs):
    return [Visualiser(path, **kwargs) for path in paths]


def get_visualisers_dict(paths, names, **kwargs):
    return {name: Visualiser(path, **kwargs) for path, name in zip(paths, names)}


def plot_risks_select_combinations(
            self, acquisition_risks, errors='std',
            fig=None, ax=None, alpha=0.3, i=0, labels=None, lw=1,
            white_bg=True, lw2=None, colors=None, acqrisk2color=None,
            skip=None, zorder=1, lwfill=1,
            ):
    """Plot mean +- std error.

    Of the risk-estimator-acquisition-function combination against the
    true empirical test risk.

    """
    if errors == 'std_error':
        middle = self.means
        sqrtN = np.sqrt(self.n_runs)
        upper_base = middle + self.stds/sqrtN
        lower_base = middle - self.stds/sqrtN
    elif errors == 'std':
        middle = self.means
        upper_base = middle + self.stds
        lower_base = middle - self.stds
    elif errors == 'percentiles':
        middle = self.means
        lower_base, upper_base = self.percentiles
    else:
        raise ValueError(f'Do not recognize errors={errors}.')
    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    linestyles = itertools.cycle(['--', '-.', ':'])
    for i, (acquisition, risk) in enumerate(acquisition_risks):
        if skip is not None and i < skip:
            continue
        acq_risk = f'{acquisition}_{risk}'
        if labels is not None:
            label = labels[i]
        else:
            label = acq_risk.replace('_', '-')
        if colors is None:
            color = resolve_color(acq_risk, acqrisk2color)
        else:
            color = colors[i]
        m = middle.loc[acquisition][risk].values
        s_u = upper_base.loc[acquisition][risk].values
        s_l = lower_base.loc[acquisition][risk].values
        x = np.arange(1, s_l.size + 1)

        if white_bg:
            ax.fill_between(
                x, s_u, s_l,
                color='white', alpha=1, zorder=zorder-10, lw=lwfill)

        ax.fill_between(
            x, s_u, s_l,
            color=color, alpha=alpha, zorder=zorder-10, lw=lwfill)
        if lw > 0:
            ax.plot(x, s_l, '--', color=color, zorder=zorder, lw=lw)
            ax.plot(x, s_u, '--', color=color, zorder=zorder, lw=lw)
        ax.plot(x, m, color=color,
                label=label, zorder=zorder, lw=lw2)
        i += 1

    return fig, ax


def plot_log_convergence(
            self, acquisition_risks, errors='std',
            fig=None, ax=None, alpha=0.3, i=0, names=None, labels=None,
            rolling=False, zorder=100, colors=None, with_errors=False,
            swapaxes=False, print_it=False, lw=None, error_type='quant',
            scale='default', lwfill=None, acqrisk2color=None, linestyle=None,
            get_alpha=None, prefac=1, transform=None, skip=None, error_fac=3,
            constantify=None,
):
    """Plot mean +- std error.

    Of the risk-estimator-acquisition-function combination against the
    true empirical test risk.

    """
    if errors == 'percentiles':
        upper_base = self.quant_errors
    elif errors == 'std':
        upper_base = self.errors
    elif errors == 'log mean':
        upper_base = self.log_sq_diff
        if scale != 'manual log':
            raise ValueError('Log target!')
    else:
        raise ValueError

    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    linestyles = itertools.cycle(['--', '-.', ':'])
    for i, (acquisition, risk) in enumerate(acquisition_risks):
        if skip is not None and i in skip:
            continue
        acq_risk = f'{acquisition}_{risk}'
        if print_it:
            print(acq_risk)

        if colors is None:
            color = resolve_color(acq_risk, acqrisk2color)
        else:
            color = colors[i]

        if linestyle is None:
            ls = '-'
        else:
            ls = linestyle(acq_risk)

        if get_alpha is None:
            alpha = 1
        else:
            alpha = get_alpha(acq_risk)

        if isinstance(zorder, list):
            zorder_i = zorder[i]
        else:
            zorder_i = zorder

        if isinstance(lw, list):
            lw_i = lw[i]
        else:
            lw_i = lw


        y = prefac * upper_base.loc[acquisition][risk].values

        if transform is not None:
            y = transform(y)

        if (R := rolling) is not False:
            y = np.convolve(
                y, np.ones(R)/R, mode='valid')

        if scale == 'manual log':
            plot = ax.plot
            if errors != 'log mean':
                y = np.log10(y)
        else:
            plot = ax.loglog

        x = np.arange(1, y.size+1)
        kwargs = dict(
            color=color, label=labels[i], zorder=zorder_i, lw=lw_i, ls=ls,
            alpha=alpha)

        if constantify is not None and i in constantify:
            y[1:] = y[0]

        if swapaxes:
            plot(y, x, **kwargs)
        else:
            plot(x, y, **kwargs)
            # print(self.cfg['dataset']['name'], acquisition, risk, y[0])

        if with_errors and errors == 'percentiles':
            low, up = self.extra_quant_errors
            low = low.loc[acquisition][risk].values * prefac
            up = up.loc[acquisition][risk].values * prefac

            if (R := rolling) is not False:
                up = np.convolve(
                    up, np.ones(R)/R, mode='valid')
                low = np.convolve(
                    low, np.ones(R)/R, mode='valid')
                x = np.arange(0, len(up))

        elif with_errors and errors == 'std':
            std = error_fac * self.std_errors.loc[acquisition][risk].values/np.sqrt(self.n_runs) * prefac
            low, up = y-std, y+std

        elif with_errors and error_type == 'std_log_error':
            middle = y
            std = self.log_sq_diff_std.loc[acquisition][risk].values
            std = std/np.sqrt(self.n_runs) * prefac
            if scale != 'manual log':
                std = np.power(std, 10)
            low = middle - std
            up = middle + std

            if with_errors and (R := rolling) is not False:
                up = np.convolve(
                    up, np.ones(R)/R, mode='valid')
                low = np.convolve(
                    low, np.ones(R)/R, mode='valid')
                x = np.arange(0, len(up))

        # elif with_errors and error_type == 'log_std_error':
        #     middle = y
        #     std = self.stds.loc[acquisition][risk].values
        #     std = std**2 / self.n_runs
        #     if add_sqrt:
        #         std = np.sqrt(std)
        #     if scale == 'manual log':
        #         std = np.log10(std)
        #     low = middle - std
        #     up = middle + std

        # else:
            # raise

        if with_errors and swapaxes:
            raise

        if constantify is not None and i in constantify:
            low[1:] = low[0]
            up[1:] = up[0]

        if with_errors:
#             ax.fill_between(x, low, up, color='white', alpha=1, zorder=-100)
            ax.fill_between(
                x, low, up, color=color, alpha=0.3, zorder=-100, lw=lwfill)
            # plot(x, std, color=color, alpha=0.3, zorder=-100, lw=1)

        i += 1

    return fig, ax


""""🔥🔥🔥 Specific Figures."""



def figure_missing7new_main(vis, errors='std'):

    fig = plt.figure(figsize=(linewidth, 0.5*linewidth), dpi=200)

    ax1 = fig.add_subplot(1, 1, 1)
    axs = [ax1]

    acq_risks  = [
        ['RandomAcquisition_NoSave', 'BiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy_Surr', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionSurrogateWeightedBALD2_SurrNoSample', 'FullSurrogateASMC'],
    ]

    colors = [RANDOM_COLOR, IS_COLOR, ASMC_COLOR]

    labels = [
        LABEL_RANDOM,
        LURE_LABEL + r' $\sim \mathbb{E}[\mathrm{Loss}]$',
        ASMC_LABEL + ' XWING',
        ]
    error_fac = 2

    plot_log_convergence(
        vis, acq_risks, errors=errors, alpha=0.5, fig=fig, ax=axs[0],
        labels=labels,
        colors=colors,
        # prefac=1/name**2,
        # transform=None,
        skip=None,
        with_errors=True,
        lwfill=0,
        lw=1.5,
        error_fac=error_fac,
        )

    # styling 🔥🔥🔥🔥
    xs = [-0.1, -0.08, -0.1, -0.08]
    for i in range(len(axs)):
        axs[i].set_xscale('linear')
        axs[i].minorticks_off()

    axs[0].legend(**new_kwargs)
    # axs[0].set_xlim(-100, 5100)
    axs[0].set_xlim(-50, 5100)
    axs[0].set_ylabel('Squared Error')
    axs[0].set_xlabel(ACQUISITION_STEP_LABEL, labelpad=3)
    axs[0].set_ylim(axs[0].get_ylim()[0], 3e0)
    # plt.setp(axs[0].get_xticklabels(), visible=False)

    schedule = list(vis.cfg['acquisition_configs']['Surr']['lazy_schedule'])
    plt_schedule = np.repeat(schedule, 2).reshape(-1, 2).T
    old_lims = plt.gca().get_ylim()
    plt_lims = np.broadcast_to(old_lims, (len(schedule), 2)).T
    axs[0].plot(plt_schedule, plt_lims, '-', color='grey', alpha=0.1, lw=.5, zorder=-10)
    axs[0].set_ylim(*old_lims)

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(
        f'notebooks/plots/missing7mainnew_{errors}.pdf',
        bbox_inches='tight',
        pad_inches=0.02)


def figure_missing7new_main_lure_and_ase_acquisitions(vis, errors='std'):

    # fig = plt.figure(figsize=(textwidth, 0.3*textwidth), dpi=200)

    # ax2 = fig.add_subplot(1, 2, 1)
    # ax1 = fig.add_subplot(1, 2, 2, sharey=ax2)

    fig, axs = plt.subplots(1, 2, figsize=(textwidth, 0.55*linewidth), dpi=200, sharey=False)

    ax0 = axs[0]
    ax1 = axs[1]
    axs = ['', '', axs[0], '', axs[1]]
    error_fac = 2


    #### 🔥🔥🔥🔥 PLOT TWO

    acq_risks  = [
        ['RandomAcquisition_NoSave', 'BiasedRiskEstimator'],
        # ['SelfSurrogateAcquisitionEntropy_Surr', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionSurrogateWeightedBALD2_SurrNoSample', 'FullSurrogateASMC'],
        ['SelfSurrogateAcquisitionSurrogateMutualInformation_SurrNoSample', 'FullSurrogateASMC'],
        ['SelfSurrogateAcquisitionEntropy_Surr', 'FullSurrogateASMC'],
        # ['SelfSurrogateAcquisitionSurrogateEntropyPlusExpectedLoss_Surr', 'FullSurrogateASMC'],
    ]

    colors = [RANDOM_COLOR, ASMC_COLOR, pal[-2], pal[1], pal[3]]

    labels = [
        LABEL_RANDOM,
        ASMC_LABEL + ' XWING',
        ASMC_LABEL + ' BALD',
        ASMC_LABEL + r' $\sim \mathbb{E}[\mathrm{Loss}]$',
        ]
    zorders = [
        -10,
        -3,
        -9,
        -8,
        -7,
    ]

    def get_style(acq_risk):
        # if 'defensive' in acq_risk.lower() and not 'asmc' in acq_risk.lower():
            # return '--'
        # else:
        # if 'NoSample' in acq_risk:
        #     return ':'

        return '-'

    lw = [1.5, 1.5, 1.5, 1.5, 1.5]

    if errors == 'percentiles':
        with_errors = False

    else:
        with_errors = True

    plot_log_convergence(
        vis, acq_risks, errors=errors, alpha=0.5, fig=fig, ax=axs[2],
        labels=labels,
        colors=colors,
        # prefac=1/name**2,
        # transform=None,
        linestyle=get_style,
        skip=None,
        with_errors=with_errors,
        lwfill=0,
        lw=lw,
        zorder=zorders,
        error_fac=error_fac,
        )

    xs = [-0.1, -0.08, -0.1, -0.08]
    for i in [2]:
        axs[i].set_xscale('linear')
        axs[i].minorticks_off()

    axs[2].legend(**new_kwargs, loc='upper right', facecolor='white',
        frameon=True)

    if errors == 'std':
        axs[2].set_ylim(.5e-4, 2e2)
    elif errors == 'percentiles':
        axs[2].set_ylim(axs[2].get_ylim()[0], 1e2)

    axs[2].set_xlim(-100, 5100)
    axs[2].set_xlabel(ACQUISITION_STEP_LABEL, labelpad=3)
    axs[2].set_ylabel('Squared Error')

    schedule = list(vis.cfg['acquisition_configs']['Surr']['lazy_schedule'])
    plt_schedule = np.repeat(schedule, 2).reshape(-1, 2).T
    old_lims = axs[2].get_ylim()
    plt_lims = np.broadcast_to(old_lims, (len(schedule), 2)).T
    axs[2].plot(plt_schedule, plt_lims, '-', color='grey', alpha=0.1, lw=.5, zorder=-10)
    axs[2].set_ylim(*old_lims)


    ### 🔥🔥🔥🔥  PLOT 4

    acq_risks  = [
        # ['RandomAcquisition_NoSave', 'FancyUnbiasedRiskEstimator'],
        ['RandomAcquisition_NoSave', 'BiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy_Surr', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionSurrogateWeightedBALD2_Surr', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionSurrogateMutualInformation_Surr', 'FancyUnbiasedRiskEstimator'],

        # ['SelfSurrogateAcquisitionSurrogateWeightedBALD2_Surr', 'FullSurrogateASMC'],
        # ['SelfSurrogateAcquisitionSurrogateEntropyPlusExpectedLoss_Surr', 'FancyUnbiasedRiskEstimator'],
    ]

    colors = [RANDOM_COLOR, IS_COLOR, pal[0], pal[-3], pal[-3]]

    labels = [
        '',
        # IS_LABEL + ' Random',
        LURE_LABEL + r' $\sim \mathbb{E}[\mathrm{Loss}]$',
        LURE_LABEL + ' $\sim$ XWING',
        LURE_LABEL + ' $\sim$ BALD',
        # LURE_LABEL + ' Exp. Loss + Unc.',
    ]


    def get_style(acq_risk):
        # if 'defensive' in acq_risk.lower() and not 'asmc' in acq_risk.lower():
            # return '--'
        # else:
        # if 'NoSample' in acq_risk:
        #     return ':'
        # if 'RandomAcquisition_NoSave_FancyUnbiasedRiskEstimator' == acq_risk:
        #     return '--'

        return '-'

    zorders = [-10, -5, -9, -8, -7]

    plot_log_convergence(
        vis, acq_risks, errors=errors, alpha=0.5, fig=fig, ax=ax1,
        labels=labels,
        colors=colors,
        # prefac=1/name**2,
        # transform=None,
        linestyle=get_style,
        skip=None,
        zorder=zorders,
        with_errors=True,
        lwfill=0,
        lw=1,
        error_fac=error_fac,
        )


    # styling 🔥🔥🔥🔥

    ax1.legend(**new_kwargs, loc='upper right', facecolor='white',
        framealpha=.8, frameon=True)

    ax1.set_xlabel(ACQUISITION_STEP_LABEL, labelpad=3)
    # ax1.set_ylim(axs[0].get_ylim()[0], 5e3)

    # axs[2].set_ylabel('Estimator Error')


    # plt.setp(axs[0].get_xticklabels(), visible=False)
    # plt.setp(axs[1].get_xticklabels(), visible=False)
    # plt.setp(axs[1].get_yticklabels(), visible=False)
    # plt.setp(axs[3].get_yticklabels(), visible=False)
    # axs[1].set_yticklabels([])
    # axs[2].set_xlabel(ACQUISITION_STEP_LABEL, labelpad=2)
    # axs[0].set_ylabel('Relative Error')
    # axs[2].legend(**new_kwargs, loc='upper right', facecolor='white',
    #     framealpha=1, frameon=True)
    # # axs[1].set_xlabel(ACQUISITION_STEP_LABEL)
    # # axs[1].xaxis.set_label_coords(1.05, -0.3)

    # schedule = list(vis.cfg['acquisition_configs']['Surr']['lazy_schedule'])
    # plt_schedule = np.repeat(schedule, 2).reshape(-1, 2).T
    # plt_lims = np.broadcast_to(plt.gca().get_ylim(), (len(schedule), 2)).T
    # ax1.plot(plt_schedule, plt_lims, '-', color='grey', alpha=0.1, lw=.5, zorder=-10)
    old_lims = ax1.get_ylim()
    plt_lims = np.broadcast_to(old_lims, (len(schedule), 2)).T
    ax1.plot(plt_schedule, plt_lims, '-', color='grey', alpha=0.1, lw=.5, zorder=-10)
    ax1.set_ylim(*old_lims)

    # styling 🔥🔥🔥🔥
    titles = ['(a)', '(a)', '(a)', '(b)', '(b)']
    xs = ['' , -0.1, -0.0, -0.1, -0.0]
    for i in [2, 4]:
        axs[i].set_xscale('linear')
        axs[i].minorticks_off()

    ax0.text(
        -0.17, 1, '(a)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax0.transAxes)

    ax1.text(
        -0.13, 1, '(b)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes)

        # axs[i].set_title(titles[i], fontsize=fs, loc='left', x=xs[i], pad=7)

    ax1.set_xlim(-100, 5100)
    ax0.set_ylim(ax0.get_ylim()[0], 5e1)
    ax1.set_ylim(*ax0.get_ylim())

    # axs[1].set_ylim(.5e-4, .5e2)
    # axs[0].set_ylim(.5e-4, .5e2)

    # for i, ax in enumerate(axs):
        # ax.set_xscale('linear')
        # ax.set_ylim(ax.get_ylim()[0], 1e1)
        # if i == 0:
            # ax.set_xlabel(ACQUISITION_STEP_LABEL, labelpad=-1)
        # else:
            # ax.set_xlabel(r'N\textsuperscript{\underline{o}} Acq\dots', labelpad=-1)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(
        f'notebooks/plots/missing7new_main_lure_and_ase_acquisitions{errors}.pdf',
        bbox_inches='tight',
        pad_inches=0.02)


def figure_resnets_unseen(viss, errors='percentiles'):

    acq1 = 'AnySurrogateAcquisitionEntropy_LazySurrEnsemble'
    acq2 = 'AnySurrogateAcquisitionEntropy_LazySurrEnsembleNoSample'
    # acq3 = 'ClassifierAcquisitionEntropy_Sample'
    acq3 = 'ClassifierAcquisitionEntropy'
    risk1 = 'FancyUnbiasedRiskEstimator'
    risk2 = 'QuadratureRiskEstimator'



    # labels = LABELS_DEFAULT_AT + ['']

    labels = [
        LABEL_RANDOM,
        LURE_LABEL + r' $\sim \mathbb{E}[\mathrm{Loss}]$',
        ASMC_LABEL + ' (const.)',
        ''
        ]

    colors = [RANDOM_COLOR, IS_COLOR, ASMC_COLOR, pal[-3]]

    zorders = [-10, -9, -2, -8]

    fig, axs = plt.subplots(
        2, 3, figsize=(textwidth, 0.4*textwidth),
        sharex=False, sharey=False, dpi=200)

    axs = np.stack([axs[:, 2], axs[:, 1], axs[:, 0]], 1)

    for i, (name, vis) in enumerate(viss.items()):

        if vis.cfg.dataset.name == 'FashionMNISTDataset':
            acq1i = acq1.replace('Ensemble', 'EnsembleLarge')
            acq2i = acq2.replace('Ensemble', 'EnsembleLarge')
        else:
            acq1i, acq2i = acq1, acq2

        acq_risks = [
            ['RandomAcquisition', 'BiasedRiskEstimator'],
            [acq1i, risk1],
            [acq1i, 'QuadratureRiskEstimator'],
            ['RandomAcquisition', 'TrueRiskEstimator']
        ]

        # 🎨🎨 TOP PLOT

        plot_risks_select_combinations(
            vis, acq_risks[:-1],
            labels=labels, errors=errors,
            colors=colors,
            fig=fig, ax=axs[0][i],
            lw=0, lw2=1, lwfill=0,
            )

        axs[0][i].set_title(name, fontsize=label_fs, pad=4)
        axs[0][i].set_xlim(1, 100)
        axs[0][i].set_xticks([1, 25, 50, 100])



        def get_style(acq_risk):
            if 'TrueRiskEstimator' in acq_risk:
                return '--'
            else:
                return '-'

        # 🎨🎨 BOTTOM PLOT
        plot_log_convergence(
            vis, acq_risks,
            errors=errors,
            labels=labels,
            linestyle=get_style,
            fig=fig, ax=axs[1][i],
            with_errors=True,
            colors=colors,
            error_fac=2,
            lwfill=0,
            constantify=[2],
            zorder=zorders,
            print_it=True,
            )

        # axs[1][i].set_xscale('linear')
        axs[0][i].set_xlim(1, 50)
        axs[1][i].set_xlim(1, 50)


    axs = np.stack([axs[:, 2], axs[:, 1], axs[:, 0]], 1)
    # for i in range(3):
        # axs[1][i].set_xscale('linear')

    # axs[0][0].set_ylim(axs[0][0].get_ylim()[0], 0.45)
    if errors == 'percentiles':
    #     axs[0][2].set_ylim(axs[0][2].get_ylim()[0], 3)
    #     axs[0][0].set_yticks([0, 0.3])
    #     # axs[1][0].set_ylim(1e-4, 6e-2)
    #     # axs[1][1].set_ylim(1e-4, 6e-2)
    #     # axs[1][2].set_ylim(1e-3, 1e0)
        axs[0][0].legend(**new_kwargs)
    else:
        axs[0][2].legend(**new_kwargs)

    # else:
    #     axs[1][2].legend(**new_kwargs)
    #     axs[1][0].set_ylim(1e-4, axs[1][0].get_ylim()[1])
    #     axs[1][1].set_ylim(1e-4, axs[1][1].get_ylim()[1])
    #     axs[1][2].set_ylim(1e-3, axs[1][2].get_ylim()[1])


    for tick in axs[1][2].yaxis.get_minor_ticks():
        tick.set_visible(False)

    axs[0][0].set_ylabel('Difference to \n True Test Risk')
    axs[1][0].set_ylabel(LABEL_STD)
    axs[0][0].set_ylim(axs[0][0].get_ylim()[0], 3)

    # axs[1][0].set_xlabel(ACQUISITION_STEP_LABEL)
    axs[1][1].set_xlabel(ACQUISITION_STEP_LABEL)
    # axs[1][2].set_xlabel(ACQUISITION_STEP_LABEL)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.35)
    plt.savefig(
        f'notebooks/plots/resnets_unseen_{errors}.pdf', bbox_inches='tight', pad_inches=0.02)
    # ax.set_ylim(1e-10, 1e0)
    # ax.set_xlim(1, 1e3)
    # ax.set_xscale('linear')


def figure_missing7_main_constant(vis, errors='std', shading=False):

    fig = plt.figure(figsize=(linewidth, 0.5*linewidth), dpi=200)

    ax1 = fig.add_subplot(1, 1, 1)
    axs = [ax1]
    error_fac = 2


    #### 🔥🔥🔥🔥 PLOT TWO


    acq_risks  = [
        ['RandomAcquisition_NoSave', 'BiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy_NoSave', 'FancyUnbiasedRiskEstimator'],
        # ['ClassifierAcquisitionEntropy_NoSave', 'FancyUnbiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy_NoSave', 'FullSurrogateASMC'],
    ]

    colors = [RANDOM_COLOR, IS_COLOR, ASMC_COLOR, ASMC_COLOR]

    labels = [
        LABEL_RANDOM,
        LURE_LABEL + ' Constant $\pi$',
        ASMC_LABEL + ' Constant $\pi$',
        ]

    def get_style(acq_risk):
        # if 'defensive' in acq_risk.lower() and not 'asmc' in acq_risk.lower():
            # return '--'
        # else:
        if 'RandomAcquisition_NoSave' not in acq_risk:
            return '--'

        return '-'

    plot_log_convergence(
        vis, acq_risks, errors=errors, alpha=0.5, fig=fig, ax=ax1,
        labels=labels,
        colors=colors,
        # prefac=1/name**2,
        # transform=None,
        linestyle=get_style,
        skip=None,
        with_errors=shading,
        lwfill=0,
        lw=1.5,
        zorder=-10,
        error_fac=error_fac,
        )

    ### 🔥🔥🔥🔥  PLOT 4

    acq_risks  = [
        ['RandomAcquisition_NoSave', 'BiasedRiskEstimator'],
        # ['RandomAcquisition_NoSave', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy_Surr', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionSurrogateEntropy_Surr', 'FancyUnbiasedRiskEstimator'],
        # ['SelfSurrogateAcquisitionSurrogateEntropyPlusExpectedLoss_Surr', 'FancyUnbiasedRiskEstimator'],
    ]



    # styling 🔥🔥🔥🔥
    for i in [0]:
        axs[i].set_xscale('linear')
        axs[i].minorticks_off()

    if shading:
        ax1.set_ylim(ax1.get_ylim()[0], 5e2)
    else:
        ax1.set_ylim(ax1.get_ylim()[0], 19e3)
    # axs[0].legend(**new_kwargs)
    ax1.legend(**new_kwargs, loc='upper right', facecolor='white',
        framealpha=.9, frameon=True)
    ax1.set_xlabel(ACQUISITION_STEP_LABEL, labelpad=3)
    ax1.set_ylabel('Squared Error')
    ax1.set_xlim(-100, 5100)

    schedule = list(vis.cfg['acquisition_configs']['Surr']['lazy_schedule'])
    plt_schedule = np.repeat(schedule, 2).reshape(-1, 2).T
    plt_lims = np.broadcast_to(plt.gca().get_ylim(), (len(schedule), 2)).T
    ax1.plot(plt_schedule, plt_lims, '-', color='grey', alpha=0.1, lw=1, zorder=-10)

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(
        f'notebooks/plots/missing7_main_constant{errors}.pdf',
        bbox_inches='tight',
        pad_inches=0.02)


def figure_missing7_main_lure_acquisitions(vis, errors='std'):

    fig = plt.figure(figsize=(linewidth, 0.5*linewidth), dpi=200)

    ax1 = fig.add_subplot(1, 1, 1)
    axs = [ax1]
    error_fac = 2


    ### 🔥🔥🔥🔥  PLOT 4

    acq_risks  = [
        ['RandomAcquisition_NoSave', 'BiasedRiskEstimator'],
        # ['RandomAcquisition_NoSave', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy_Surr', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionSurrogateEntropy_Surr', 'FancyUnbiasedRiskEstimator'],
        # ['SelfSurrogateAcquisitionSurrogateEntropyPlusExpectedLoss_Surr', 'FancyUnbiasedRiskEstimator'],
    ]

    colors = [RANDOM_COLOR, IS_COLOR, pal[-2], pal[-3], pal[-3]]

    labels = [
        LABEL_RANDOM,
        # IS_LABEL + ' Random',
        LURE_LABEL + ' Expected Loss',
        LURE_LABEL + ' Uncertainty',
        LURE_LABEL + ' Exp. Loss + Unc.',]


    def get_style(acq_risk):
        # if 'defensive' in acq_risk.lower() and not 'asmc' in acq_risk.lower():
            # return '--'
        # else:
        if 'NoSample' in acq_risk:
            return ':'
        if 'RandomAcquisition_NoSave_FancyUnbiasedRiskEstimator' == acq_risk:
            return '--'

        return '-'

    zorders = [-10, -5, -9, -8, -7]

    plot_log_convergence(
        vis, acq_risks, errors=errors, alpha=0.5, fig=fig, ax=ax1,
        labels=labels,
        colors=colors,
        # prefac=1/name**2,
        # transform=None,
        linestyle=get_style,
        skip=None,
        zorder=zorders,
        with_errors=True,
        lwfill=0,
        lw=1,
        error_fac=error_fac,
        )


    # styling 🔥🔥🔥🔥
    for i in [0]:
        axs[i].set_xscale('linear')
        axs[i].minorticks_off()

    ax1.legend(**new_kwargs, loc='upper left', facecolor='white',
        framealpha=.7, frameon=True)

    ax1.set_xlabel(ACQUISITION_STEP_LABEL, labelpad=3)
    ax1.set_ylabel('Squared Error')
    ax1.set_xlim(-100, 5100)
    ax1.set_ylim(axs[0].get_ylim()[0], 5e3)

    # axs[2].set_ylabel('Estimator Error')


    # plt.setp(axs[0].get_xticklabels(), visible=False)
    # plt.setp(axs[1].get_xticklabels(), visible=False)
    # plt.setp(axs[1].get_yticklabels(), visible=False)
    # plt.setp(axs[3].get_yticklabels(), visible=False)
    # axs[1].set_yticklabels([])
    # axs[2].set_xlabel(ACQUISITION_STEP_LABEL, labelpad=2)
    # axs[0].set_ylabel('Relative Error')
    # axs[2].legend(**new_kwargs, loc='upper right', facecolor='white',
    #     framealpha=1, frameon=True)
    # # axs[1].set_xlabel(ACQUISITION_STEP_LABEL)
    # # axs[1].xaxis.set_label_coords(1.05, -0.3)

    schedule = list(vis.cfg['acquisition_configs']['Surr']['lazy_schedule'])
    plt_schedule = np.repeat(schedule, 2).reshape(-1, 2).T
    plt_lims = np.broadcast_to(plt.gca().get_ylim(), (len(schedule), 2)).T
    ax1.plot(plt_schedule, plt_lims, '-', color='grey', alpha=0.1, lw=1, zorder=-10)

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(
        f'notebooks/plots/missing7_main_lure_acquisitions{errors}.pdf',
        bbox_inches='tight',
        pad_inches=0.02)


def figure_missing7new_acquisitions_sampling(vis, errors='std'):

    fig = plt.figure(figsize=(linewidth, 0.5*linewidth), dpi=200)

    ax2 = fig.add_subplot(1, 1, 1)
    axs = ['', '', ax2, '']
    error_fac = 2
    ### 🔥🔥🔥🔥  PLOT 3

    acq_risks  = [
        ['RandomAcquisition_NoSave', 'BiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionSurrogateWeightedBALD2_SurrNoSample', 'FullSurrogateASMC'],
        ['SelfSurrogateAcquisitionSurrogateWeightedBALD2_Surr', 'FullSurrogateASMC'],
        ['SelfSurrogateAcquisitionSurrogateMutualInformation_Surr', 'FullSurrogateASMC'],
        # ['SelfSurrogateAcquisitionSurrogateEntropy_Surr', 'FullSurrogateASMC'],
        # ['SelfSurrogateAcquisitionSurrogateMutualInformation_Surr', 'FullSurrogateASMC'],
        # ['SelfSurrogateAcquisitionSurrogateEntropyPlusExpectedLoss_Surr', 'FullSurrogateASMC'],
    ]

    colors = [RANDOM_COLOR, ASMC_COLOR, pal[-2], pal[1], pal[3]]

    labels = [
        LABEL_RANDOM,
        ASMC_LABEL + ' XWING',
        ASMC_LABEL + ' $\sim$ XWING',
        ASMC_LABEL + ' $\sim$ BALD',
        ASMC_LABEL + ' Expected Loss (det)',
        ]

    zorders = [
        -10,
        -3,
        -9,
        -8,
        -7,
    ]

    def get_style(acq_risk):
        # if 'defensive' in acq_risk.lower() and not 'asmc' in acq_risk.lower():
            # return '--'
        # else:
        # if 'NoSample' in acq_risk:
        #     return ':'

        return '-'

    lw = [1.5, 1.5, 1.5, 1.5, 1.5]

    if errors == 'percentiles':
        with_errors = False

    else:
        with_errors = True

    plot_log_convergence(
        vis, acq_risks, errors=errors, alpha=0.5, fig=fig, ax=axs[2],
        labels=labels,
        colors=colors,
        # prefac=1/name**2,
        # transform=None,
        linestyle=get_style,
        skip=None,
        with_errors=with_errors,
        lwfill=0,
        lw=lw,
        zorder=zorders,
        error_fac=error_fac,
        )

    # styling 🔥🔥🔥🔥
    xs = [-0.1, -0.08, -0.1, -0.08]
    for i in [2]:
        axs[i].set_xscale('linear')
        axs[i].minorticks_off()

    axs[2].legend(**new_kwargs, loc='upper right', facecolor='white',
        frameon=True)

    if errors == 'std':
        axs[2].set_ylim(.5e-4, 2e2)
    elif errors == 'percentiles':
        axs[2].set_ylim(axs[2].get_ylim()[0], 1e2)

    axs[2].set_xlim(-100, 5100)

    axs[2].set_xlabel(ACQUISITION_STEP_LABEL, labelpad=3)
    axs[2].set_ylabel('Squared Error')

    schedule = list(vis.cfg['acquisition_configs']['Surr']['lazy_schedule'])
    plt_schedule = np.repeat(schedule, 2).reshape(-1, 2).T
    old_lims = plt.gca().get_ylim()
    plt_lims = np.broadcast_to(old_lims, (len(schedule), 2)).T
    axs[2].plot(plt_schedule, plt_lims, '-', color='grey', alpha=0.1, lw=.5, zorder=-10)
    axs[2].set_ylim(*old_lims)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(
        f'notebooks/plots/missing7new_acquisitions_sampling_{errors}.pdf',
        bbox_inches='tight',
        pad_inches=0.02)


