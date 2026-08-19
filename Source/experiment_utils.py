"""Experiment logging and dataset split diagnostics.

Two groups of utilities supporting the training pipeline:

* Experiment logging — ExperimentLogger, which maintains one main log file per
  experiment plus one per cross-validation fold, and setup_simple_logger for a
  single timestamped log file.
* Split diagnostics — feature accessors shared by the reporting routines,
  per-subset summary statistics exported as CSV/JSON tables, and the
  quality-control figures used to verify that redox potential, molecular weight,
  solvent identity, data type and redox type are balanced across folds.

The splitting routines in data.py import save_split_statistics and
save_split_plots from here; this module does not import data.py.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

PALETTE = {
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#55A868",
    "calc": "#8172B3",
    "exp": "#C44E52",
    "ox": "#64B5CD",
    "red": "#E77C8E",
}
FONT_SIZE = 16


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Manages loggers for an experiment: one main logger and one per fold."""

    def __init__(self, experiment_name: str, log_dir: str = "logs", console_level=logging.INFO):
        """
        Args:
            experiment_name: Experiment identifier used for directory and logger names.
            log_dir: Root directory for log files.
            console_level: Logging level for console output.
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.main_logger = self._setup_logger(
            name=f"{experiment_name}_main",
            log_file=self.log_dir / "training.log",
            console_level=console_level,
        )
        self.fold_loggers = {}

    def _setup_logger(
        self,
        name: str,
        log_file: Path,
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    ) -> logging.Logger:
        """Create a logger writing to both a file and stdout."""
        experiment_logger = logging.getLogger(name)
        experiment_logger.setLevel(logging.DEBUG)
        experiment_logger.handlers.clear()

        file_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )

        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(file_formatter)
        experiment_logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(console_formatter)
        experiment_logger.addHandler(ch)

        return experiment_logger

    def get_fold_logger(self, fold_num: int) -> logging.Logger:
        """Return (creating if necessary) the logger for a given fold."""
        if fold_num not in self.fold_loggers:
            self.fold_loggers[fold_num] = self._setup_logger(
                name=f"{self.experiment_name}_fold_{fold_num}",
                log_file=self.log_dir / f"fold_{fold_num}.log",
            )
        return self.fold_loggers[fold_num]

    def log_experiment_start(self, config: dict):
        """Log experiment start and full configuration."""
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"Starting experiment: {self.experiment_name}")
        self.main_logger.info("Configuration:")
        for key, value in config.items():
            self.main_logger.info(f"  {key}: {value}")
        self.main_logger.info("-" * 80)

    def log_fold_start(self, fold_num: int, train_size: int, val_size: int):
        """Log fold start with dataset sizes."""
        fold_logger = self.get_fold_logger(fold_num)
        fold_logger.info(f"Fold {fold_num} — train: {train_size}, val: {val_size}")
        self.main_logger.info(f"Starting Fold {fold_num} (train: {train_size}, val: {val_size})")

    def log_fold_metrics(self, fold_num: int, metrics: dict, phase: str = "validation"):
        """Log per-fold metrics for a given phase."""
        fold_logger = self.get_fold_logger(fold_num)
        fold_logger.info(f"{phase.capitalize()} metrics:")
        for name, value in metrics.items():
            if phase in name.lower():
                fold_logger.info(f"  {name}: {value:.4f}")
        self.main_logger.info(f"Fold {fold_num} {phase} completed")

    def log_experiment_summary(self, summary: dict):
        """Log final experiment summary."""
        self.main_logger.info("=" * 80)
        self.main_logger.info("Experiment summary")
        for key, value in summary.items():
            if isinstance(value, dict):
                self.main_logger.info(f"{key}:")
                for subkey, subvalue in value.items():
                    fmt = f"{subvalue:.4f}" if isinstance(subvalue, float) else str(subvalue)
                    self.main_logger.info(f"  {subkey}: {fmt}")
            else:
                self.main_logger.info(f"{key}: {value}")
        self.main_logger.info("=" * 80)

    def close(self):
        """Close all file handlers."""
        for experiment_logger in [self.main_logger, *self.fold_loggers.values()]:
            for handler in experiment_logger.handlers[:]:
                handler.close()
                experiment_logger.removeHandler(handler)


def setup_simple_logger(
    name: str = "training",
    log_dir: str = "logs",
    level=logging.INFO,
) -> logging.Logger:
    """Create a lightweight logger writing to a timestamped file and stdout.

    Args:
        name: Logger name and file prefix.
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    simple_logger = logging.getLogger(name)
    simple_logger.setLevel(level)
    simple_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    simple_logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    simple_logger.addHandler(ch)

    return simple_logger


# ---------------------------------------------------------------------------
# Feature accessors
# ---------------------------------------------------------------------------

def _get_mw(dp) -> float:
    """Return the molecular weight of a data object, or 0.0 if unavailable."""
    smiles = getattr(dp, "smiles", None)
    if smiles is None:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol is not None else 0.0


def _get_e_value(dp) -> float:
    """Return the scalar redox potential stored in a data object."""
    y = getattr(dp, "y", None)
    if y is None:
        return 0.0
    if isinstance(y, dict):
        tensor = y.get("E")
        return float(tensor.reshape(-1)[0]) if tensor is not None else 0.0
    return float(y.reshape(-1)[0])

def _quantile_bin_labels(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Assign each value to an equal-frequency (quantile) bin in [0, n_bins).
    All-identical arrays receive all-zero labels.
    Duplicate bin edges are collapsed; effective bin count may be < n_bins.
    """
    if np.all(values == values[0]):
        return np.zeros(len(values), dtype=int)
    edges = np.unique(np.percentile(values, np.linspace(0, 100, n_bins + 1)))
    return np.digitize(values, edges[1:-1]).astype(int)

def _composite_labels(dataset: list, n_bins_mw: int = 4, n_bins_e: int = 4) -> list[str]:
    """
    Build a composite stratification label for every sample:
        "<solvent_smiles>|mw<MW-bin>|e<E-bin>|<redox_type>"
    """
    mw_vals = np.array([_get_mw(dp) for dp in dataset], dtype=float)
    e_vals = np.array([_get_e_value(dp) for dp in dataset], dtype=float)
    mw_bins = _quantile_bin_labels(mw_vals, n_bins_mw)
    e_bins = _quantile_bin_labels(e_vals, n_bins_e)

    return [
        f"{getattr(dp, 'solvent_smiles', 'unk')}|mw{mwb}|e{eb}|{getattr(dp, 'redox_type', 'unk')}"
        for dp, mwb, eb in zip(dataset, mw_bins, e_bins)
    ]


def _merge_rare_labels(labels: list[str], min_count: int) -> list[str]:
    """Merge labels with fewer than min_count occurrences into '_rare_'."""
    counts = Counter(labels)
    return [lbl if counts[lbl] >= min_count else "_rare_" for lbl in labels]

def only_exp(subset: list) -> list:
    """Keep experimental records only."""
    return [dp for dp in subset if getattr(dp, "data_type", "") == "exp"]


FEATURES = [
    ("E-value", _get_e_value, "V", "e"),
    ("Molecular weight", _get_mw, "Da", "mw"),
]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

@dataclass
class SplitStats:
    """Numeric summary of one data subset (train, val or test)."""
    name: str
    n_total: int = 0
    n_calc: int = 0
    n_exp: int = 0
    n_ox: int = 0
    n_red: int = 0
    n_solvents: int = 0
    e_min: float = 0.0
    e_max: float = 0.0
    e_mean: float = 0.0
    e_std: float = 0.0
    e_median: float = 0.0
    mw_min: float = 0.0
    mw_max: float = 0.0
    mw_mean: float = 0.0
    mw_std: float = 0.0
    mw_median: float = 0.0


def _compute_stats(name: str, subset: list) -> SplitStats:
    if not subset:
        return SplitStats(name=name)
    e_vals = np.array([_get_e_value(dp) for dp in subset])
    mw_vals = np.array([_get_mw(dp) for dp in subset])
    return SplitStats(
        name=name,
        n_total=len(subset),
        n_calc=sum(1 for dp in subset if getattr(dp, "data_type", "") == "calc"),
        n_exp=sum(1 for dp in subset if getattr(dp, "data_type", "") == "exp"),
        n_ox=sum(1 for dp in subset if getattr(dp, "redox_type", "") == "ox"),
        n_red=sum(1 for dp in subset if getattr(dp, "redox_type", "") == "red"),
        n_solvents=len({getattr(dp, "solvent_smiles", None) for dp in subset}),
        e_min=float(e_vals.min()),
        e_max=float(e_vals.max()),
        e_mean=float(e_vals.mean()),
        e_std=float(e_vals.std()),
        e_median=float(np.median(e_vals)),
        mw_min=float(mw_vals.min()),
        mw_max=float(mw_vals.max()),
        mw_mean=float(mw_vals.mean()),
        mw_std=float(mw_vals.std()),
        mw_median=float(np.median(mw_vals)),
    )


def _solvent_counts_table(folds_data: list[tuple[list, list]], test_subset: list) -> pd.DataFrame:
    """Per-solvent sample counts across train, val and test."""
    all_solvents: set[str] = set()
    for train, val in folds_data:
        for dp in train + val:
            all_solvents.add(getattr(dp, "solvent_smiles", "unk"))
    for dp in test_subset:
        all_solvents.add(getattr(dp, "solvent_smiles", "unk"))

    n_folds = max(len(folds_data), 1)
    rows = []
    for solvent in sorted(all_solvents):
        train_n = sum(
            sum(1 for dp in tr if getattr(dp, "solvent_smiles", "") == solvent)
            for tr, _ in folds_data
        ) / n_folds
        val_n = sum(
            sum(1 for dp in vl if getattr(dp, "solvent_smiles", "") == solvent)
            for _, vl in folds_data
        ) / n_folds
        test_n = sum(1 for dp in test_subset if getattr(dp, "solvent_smiles", "") == solvent)
        rows.append({
            "solvent": solvent,
            "avg_train_N": round(train_n, 2),
            "avg_val_N": round(val_n, 2),
            "test_N": test_n,
            "total": round(train_n + val_n + test_n, 2),
        })
    return pd.DataFrame(rows)


def _sample_level_table(subset: list, split_label: str) -> pd.DataFrame:
    """Sample-level table of one subset."""
    return pd.DataFrame([{
        "split": split_label,
        "smiles": getattr(dp, "smiles", ""),
        "solvent_smiles": getattr(dp, "solvent_smiles", ""),
        "redox_type": getattr(dp, "redox_type", ""),
        "data_type": getattr(dp, "data_type", ""),
        "E_value": _get_e_value(dp),
        "MW": _get_mw(dp),
    } for dp in subset])


def _stats_block(
    folds_data: list[tuple[list, list]],
    test_subset: list,
    filter_fn=None,
    label_suffix: str = "",
) -> tuple[list[dict], dict, list[SplitStats]]:
    """Compute per-fold and test statistics, optionally filtering each subset."""
    fn = filter_fn if filter_fn is not None else (lambda x: x)
    rows: list[dict] = []
    stats_list: list[SplitStats] = []
    summary_folds = []

    for fold_idx, (train, val) in enumerate(folds_data, start=1):
        st_train = _compute_stats(f"fold_{fold_idx}_train{label_suffix}", fn(train))
        st_val = _compute_stats(f"fold_{fold_idx}_val{label_suffix}", fn(val))
        stats_list.extend([st_train, st_val])
        rows.append({**asdict(st_train), "fold": fold_idx, "split": "train"})
        rows.append({**asdict(st_val), "fold": fold_idx, "split": "val"})
        summary_folds.append({"fold": fold_idx, "train": asdict(st_train), "val": asdict(st_val)})

    st_test = _compute_stats(f"test{label_suffix}", fn(test_subset))
    stats_list.append(st_test)
    rows.append({**asdict(st_test), "fold": -1, "split": "test"})

    summary = {"n_folds": len(folds_data), "folds": summary_folds, "test": asdict(st_test)}
    return rows, summary, stats_list


def _save_statistics(
    folds_data: list[tuple[list, list]],
    test_subset: list,
    output_dir: Path,
) -> tuple[list[SplitStats], list[SplitStats]]:
    """Write split statistics for the full data and for the experimental subset.

    Files written: split_statistics_all.csv, split_statistics_exp_only.csv,
    split_summary.json, solvent_counts.csv, fold_<n>_details.csv, test_details.csv.

    Returns:
        Summary statistics of the full data and of the experimental subset,
        as consumed by save_split_plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_all, summary_all, stats_all = _stats_block(folds_data, test_subset)
    pd.DataFrame(rows_all).to_csv(output_dir / "split_statistics_all.csv", index=False)

    rows_exp, summary_exp, stats_exp = _stats_block(
        folds_data, test_subset, filter_fn=only_exp, label_suffix="_exp_only"
    )
    pd.DataFrame(rows_exp).to_csv(output_dir / "split_statistics_exp_only.csv", index=False)

    with open(output_dir / "split_summary.json", "w") as f:
        json.dump({"all": summary_all, "exp_only": summary_exp}, f, indent=2)

    _solvent_counts_table(folds_data, test_subset).to_csv(
        output_dir / "solvent_counts.csv", index=False
    )

    for fold_idx, (train, val) in enumerate(folds_data, start=1):
        pd.concat([
            _sample_level_table(train, f"fold_{fold_idx}_train"),
            _sample_level_table(val, f"fold_{fold_idx}_val"),
        ], ignore_index=True).to_csv(output_dir / f"fold_{fold_idx}_details.csv", index=False)

    _sample_level_table(test_subset, "test").to_csv(output_dir / "test_details.csv", index=False)
    logger.info("Statistics saved to %s", output_dir)
    return stats_all, stats_exp


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _save_plots(
    folds_data: list[tuple[list, list]],
    test_subset: list,
    plots_dir: Path,
    all_stats: list[SplitStats],
    filter_fn=None,
    title_suffix: str = "",
) -> None:
    """Generate the diagnostic plots of one split and save them to plots_dir.

    Args:
        folds_data: per-fold (train, val) record lists.
        test_subset: records of the held-out test set.
        plots_dir: directory for the figures, created if absent.
        all_stats: summary statistics rendered in the dashboard table.
        filter_fn: optional filter applied to every subset before plotting,
            for example only_exp to restrict the figures to experimental data.
        title_suffix: text appended to every figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    fn = filter_fn if filter_fn is not None else (lambda x: x)
    folds_data = [(fn(tr), fn(vl)) for tr, vl in folds_data]
    test_subset = fn(test_subset)
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_folds = len(folds_data)

    def subset_counts(attribute: str, value: str) -> list[int]:
        """Counts of records with attribute == value in train, val and test subsets."""
        return (
            [sum(1 for dp in tr if getattr(dp, attribute, "") == value) for tr, _ in folds_data]
            + [sum(1 for dp in vl if getattr(dp, attribute, "") == value) for _, vl in folds_data]
            + [sum(1 for dp in test_subset if getattr(dp, attribute, "") == value)]
        )

    # Per-fold distributions of E-value and molecular weight
    for feature, getter, unit, prefix in FEATURES:
        fig, axes = plt.subplots(1, n_folds, figsize=(4.5 * n_folds, 4.5))
        axes = np.atleast_1d(axes)
        for fold_idx, (train, val) in enumerate(folds_data):
            ax = axes[fold_idx]
            for subset, label, color in [
                (train, "Train", PALETTE["train"]),
                (val, "Val", PALETTE["val"]),
                (test_subset, "Test", PALETTE["test"]),
            ]:
                arr = np.array([getter(dp) for dp in subset])
                if len(arr) >= 2:
                    ax.hist(arr, bins=30, alpha=0.45, color=color, label=label, density=True)
                    ax.axvline(arr.mean(), color=color, linewidth=1.5, linestyle="--")
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=FONT_SIZE)
            ax.set_xlabel(f"{feature} ({unit})", fontsize=FONT_SIZE)
            ax.set_ylabel("Density", fontsize=FONT_SIZE)
            ax.tick_params(labelsize=FONT_SIZE - 2)
            if fold_idx == 0:
                ax.legend(fontsize=FONT_SIZE - 2)
        fig.suptitle(f"{feature} distribution per fold{title_suffix}",
                     fontsize=FONT_SIZE + 2, y=1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix}_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Stacked bars: data type balance and redox type balance
    fold_labels = (
        [f"Fold {i + 1}\nTrain" for i in range(n_folds)]
        + [f"Fold {i + 1}\nVal" for i in range(n_folds)]
        + ["Test"]
    )
    x = np.arange(len(fold_labels))

    for attribute, (val_a, val_b), (lbl_a, lbl_b), fname, title in [
        ("data_type", ("calc", "exp"), ("Calc", "Exp"),
         "fold_balance.png", "Calc / Exp balance per fold"),
        ("redox_type", ("ox", "red"), ("Oxidation", "Reduction"),
         "redox_balance.png", "Oxidation / Reduction balance per fold"),
    ]:
        counts_a = subset_counts(attribute, val_a)
        counts_b = subset_counts(attribute, val_b)
        fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(fold_labels)), 5.5))
        bars_a = ax.bar(x, counts_a, label=lbl_a, color=PALETTE[val_a], alpha=0.85)
        bars_b = ax.bar(x, counts_b, bottom=counts_a, label=lbl_b, color=PALETTE[val_b], alpha=0.85)
        for bar, n in zip(list(bars_a) + list(bars_b), counts_a + counts_b):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(n), ha="center", va="center",
                        fontsize=FONT_SIZE - 4, color="white", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, fontsize=FONT_SIZE - 3)
        ax.set_ylabel("Sample count", fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE + 2)
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.legend(fontsize=FONT_SIZE - 2)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Box plots of E-value and molecular weight
    box_labels = (
        [f"F{i + 1}\nTrain" for i in range(n_folds)]
        + [f"F{i + 1}\nVal" for i in range(n_folds)]
        + ["Test"]
    )
    box_colors = [PALETTE["train"]] * n_folds + [PALETTE["val"]] * n_folds + [PALETTE["test"]]
    legend_handles = [
        mpatches.Patch(color=PALETTE["train"], label="Train"),
        mpatches.Patch(color=PALETTE["val"], label="Val"),
        mpatches.Patch(color=PALETTE["test"], label="Test"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (feature, getter, unit, _) in zip(axes, FEATURES):
        box_data = (
            [np.array([getter(dp) for dp in tr]) for tr, _ in folds_data]
            + [np.array([getter(dp) for dp in vl]) for _, vl in folds_data]
            + [np.array([getter(dp) for dp in test_subset])]
        )
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(f"{feature} ({unit})", fontsize=FONT_SIZE)
        ax.set_title(f"{feature} box plots per fold", fontsize=FONT_SIZE + 2)
        ax.tick_params(labelsize=FONT_SIZE - 3)
        ax.legend(handles=legend_handles, fontsize=FONT_SIZE - 2)
    fig.tight_layout()
    fig.savefig(plots_dir / "e_mw_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Solvent coverage across splits
    train_exp = [dp for tr, _ in folds_data for dp in tr if getattr(dp, "data_type", "") == "exp"]
    val_exp = [dp for _, vl in folds_data for dp in vl if getattr(dp, "data_type", "") == "exp"]
    sv_train = Counter(getattr(dp, "solvent_smiles", "unk") for dp in train_exp)
    sv_val = Counter(getattr(dp, "solvent_smiles", "unk") for dp in val_exp)
    sv_test = Counter(getattr(dp, "solvent_smiles", "unk") for dp in test_subset)
    top_solvents = [s for s, _ in (sv_train + sv_val + sv_test).most_common(30)]
    train_vals = np.array([sv_train.get(s, 0) / max(n_folds, 1) for s in top_solvents])
    val_vals = np.array([sv_val.get(s, 0) / max(n_folds, 1) for s in top_solvents])
    test_vals = np.array([sv_test.get(s, 0) for s in top_solvents])
    y = np.arange(len(top_solvents))

    fig, ax = plt.subplots(figsize=(14, max(6, len(top_solvents) * 0.45)))
    ax.barh(y, train_vals, label="Average train", color=PALETTE["train"], alpha=0.85)
    ax.barh(y, val_vals, left=train_vals, label="Average val", color=PALETTE["val"], alpha=0.85)
    ax.barh(y, test_vals, left=train_vals + val_vals, label="Test",
            color=PALETTE["test"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([s if len(s) <= 20 else s[:18] + "..." for s in top_solvents],
                       fontsize=FONT_SIZE - 4)
    ax.set_xlabel("Sample count (train and val averaged over folds)", fontsize=FONT_SIZE)
    ax.set_title("Solvent coverage across splits (30 most frequent)", fontsize=FONT_SIZE + 2)
    ax.tick_params(labelsize=FONT_SIZE - 2)
    ax.legend(fontsize=FONT_SIZE - 2)
    fig.tight_layout()
    fig.savefig(plots_dir / "solvent_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Empirical CDFs: overlapping curves indicate well balanced folds
    for feature, getter, unit, prefix in FEATURES:
        fig, axes = plt.subplots(1, n_folds, figsize=(4.5 * n_folds, 4.5), sharey=True)
        axes = np.atleast_1d(axes)
        for fold_idx, (train, val) in enumerate(folds_data):
            ax = axes[fold_idx]
            for subset, label, color in [
                (train, "Train", PALETTE["train"]),
                (val, "Val", PALETTE["val"]),
                (test_subset, "Test", PALETTE["test"]),
            ]:
                arr = np.sort([getter(dp) for dp in subset])
                if len(arr) > 0:
                    ax.plot(arr, np.arange(1, len(arr) + 1) / len(arr),
                            color=color, label=label, linewidth=1.5, alpha=0.85)
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=FONT_SIZE)
            ax.set_xlabel(f"{feature} ({unit})", fontsize=FONT_SIZE)
            ax.set_ylabel("ECDF", fontsize=FONT_SIZE)
            ax.tick_params(labelsize=FONT_SIZE - 2)
            ax.grid(True, alpha=0.3)
            if fold_idx == 0:
                ax.legend(fontsize=FONT_SIZE - 2)
        fig.suptitle(f"Empirical CDF of {feature} (stratification check){title_suffix}",
                     fontsize=FONT_SIZE + 2, y=1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix}_ecdf.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary dashboard
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for col, (feature, getter, unit, _) in enumerate(FEATURES):
        ax = fig.add_subplot(gs[0, col])
        for fold_idx, (train, val) in enumerate(folds_data):
            arr = np.array([getter(dp) for dp in train + val])
            if len(arr) >= 2:
                ax.hist(arr, bins=25, alpha=0.3, density=True, label=f"F{fold_idx + 1} train+val")
        if test_subset:
            arr = np.array([getter(dp) for dp in test_subset])
            ax.hist(arr, bins=25, alpha=0.6, color=PALETTE["test"], density=True,
                    label="Test", linewidth=1.5, histtype="step")
        ax.set_xlabel(f"{feature} ({unit})", fontsize=FONT_SIZE)
        ax.set_ylabel("Density", fontsize=FONT_SIZE)
        ax.set_title(f"{feature} overlap (all folds)", fontsize=FONT_SIZE + 1)
        ax.tick_params(labelsize=FONT_SIZE - 3)
        ax.legend(fontsize=FONT_SIZE - 5)

    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=[[s.name, s.n_total, s.n_calc, s.n_exp, s.n_ox, s.n_red, s.n_solvents]
                  for s in all_stats],
        colLabels=["Subset", "N", "Calc", "Exp", "Ox", "Red", "Solvents"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE - 6)
    table.scale(1, 1.4)
    ax_table.set_title("Split counts", fontsize=FONT_SIZE + 1, pad=8)

    x_train = np.arange(n_folds)
    x_val = x_train + n_folds + 0.5
    names_train = [f"F{i + 1}\nTrain" for i in range(n_folds)]
    names_val = [f"F{i + 1}\nVal" for i in range(n_folds)]

    for col, (feature, getter, unit, _) in enumerate(FEATURES):
        ax = fig.add_subplot(gs[1, col])
        train_mean = [np.mean([getter(dp) for dp in tr]) for tr, _ in folds_data]
        train_std = [np.std([getter(dp) for dp in tr]) for tr, _ in folds_data]
        val_mean = [np.mean([getter(dp) for dp in vl]) for _, vl in folds_data]
        val_std = [np.std([getter(dp) for dp in vl]) for _, vl in folds_data]
        ax.errorbar(x_train, train_mean, yerr=train_std, fmt="o", color=PALETTE["train"],
                    capsize=4, linewidth=1.5, label="Train")
        ax.errorbar(x_val, val_mean, yerr=val_std, fmt="s", color=PALETTE["val"],
                    capsize=4, linewidth=1.5, label="Val")
        if test_subset:
            test_mean = np.mean([getter(dp) for dp in test_subset])
            ax.axhline(test_mean, color=PALETTE["test"], linestyle="--", linewidth=1.5,
                       label=f"Test mean = {test_mean:.2f}")
        ax.set_xticks(list(x_train) + list(x_val))
        ax.set_xticklabels(names_train + names_val, fontsize=FONT_SIZE - 5)
        ax.set_ylabel(f"{feature} ({unit})", fontsize=FONT_SIZE)
        ax.set_title(f"{feature} mean and standard deviation per fold", fontsize=FONT_SIZE + 1)
        ax.tick_params(labelsize=FONT_SIZE - 3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FONT_SIZE - 5)

    ax_bar = fig.add_subplot(gs[1, 2])
    calc_counts = subset_counts("data_type", "calc")
    exp_counts = subset_counts("data_type", "exp")
    x_bar = np.arange(len(fold_labels))
    ax_bar.bar(x_bar, calc_counts, label="Calc", color=PALETTE["calc"], alpha=0.85)
    ax_bar.bar(x_bar, exp_counts, bottom=calc_counts, label="Exp",
               color=PALETTE["exp"], alpha=0.85)
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(names_train + names_val + ["Test"], fontsize=FONT_SIZE - 5)
    ax_bar.set_ylabel("Sample count", fontsize=FONT_SIZE)
    ax_bar.set_title("Calc / Exp per fold", fontsize=FONT_SIZE + 1)
    ax_bar.tick_params(labelsize=FONT_SIZE - 3)
    ax_bar.legend(fontsize=FONT_SIZE - 5)

    fig.suptitle(f"Split quality control dashboard{title_suffix}",
                 fontsize=FONT_SIZE + 4, fontweight="bold")
    fig.savefig(plots_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Plots saved to %s", plots_dir)


def _only_exp(subset: list) -> list:
    return [dp for dp in subset if getattr(dp, "data_type", "") == "exp"]

def _collect_stats(dataset) -> tuple[Counter, Counter]:
    """Return (solvent_counts, redox_counts) for a dataset."""
    solvent_counts: Counter = Counter()
    redox_counts: Counter = Counter()
    for sample in dataset:
        if hasattr(sample, "solvent_smiles"):
            solvent_counts[sample.solvent_smiles] += 1
        if hasattr(sample, "redox_type"):
            redox_counts[sample.redox_type] += 1
    return solvent_counts, redox_counts


def log_dataset_composition(dataset, exp_logger: ExperimentLogger) -> None:
    """Log total size, unique solvent count, redox distribution, and top-10 solvents."""
    log = exp_logger.main_logger
    total = len(dataset)
    solvent_counts, redox_counts = _collect_stats(dataset)
    log.info("=" * 80)
    log.info("Dataset: %d samples | %d unique solvents | redox: %s",
             total, len(solvent_counts), dict(redox_counts))
    log.info("Top-10 solvents:")
    for smiles, cnt in solvent_counts.most_common(10):
        log.info("  %-50s  %4d  (%.1f%%)", smiles, cnt, cnt / total * 100)


def collect_fold_statistics(folds, test_loader) -> list[dict]:
    """
    Build a list of statistic dicts, one for the test set followed by one per fold.
    Each fold dict includes solvent overlap between train and val subsets.
    """
    stats = []

    sc, rc = _collect_stats(test_loader.dataset)
    stats.append({
        "name": "Test Set",
        "solvent_counts": sc,
        "redox_counts": rc,
        "total": len(test_loader.dataset),
        "unique_solvents": len(sc),
    })

    for i, (train_loader, val_loader) in enumerate(folds):
        train_sc, train_rc = _collect_stats(train_loader.dataset)
        val_sc, val_rc = _collect_stats(val_loader.dataset)
        stats.append({
            "name": f"Fold {i + 1}",
            "train_solvent_counts": train_sc,
            "val_solvent_counts": val_sc,
            "train_redox_counts": train_rc,
            "val_redox_counts": val_rc,
            "train_total": len(train_loader.dataset),
            "val_total": len(val_loader.dataset),
            "total": len(train_loader.dataset) + len(val_loader.dataset),
            "unique_solvents": len(train_sc + val_sc),
            "train_unique_solvents": len(train_sc),
            "val_unique_solvents": len(val_sc),
            "common_solvents": set(train_sc) & set(val_sc),
            "train_only_solvents": set(train_sc) - set(val_sc),
            "val_only_solvents": set(val_sc) - set(train_sc),
        })

    return stats


def log_fold_statistics(exp_logger: ExperimentLogger, statistics: list[dict]) -> None:
    """Log per-fold solvent and redox-type distribution to the experiment logger."""
    log = exp_logger.main_logger
    log.info("=" * 80)
    log.info("Solvent distribution")
    log.info("=" * 80)

    test = statistics[0]
    log.info("%s: total=%d | unique solvents=%d | redox=%s",
             test["name"], test["total"], test["unique_solvents"], dict(test["redox_counts"]))

    for fold in statistics[1:]:
        log.info("%s: total=%d  train=%d  val=%d", fold["name"],
                 fold["total"], fold["train_total"], fold["val_total"])
        log.info("  Solvents: total=%d  train=%d  val=%d  common=%d  val-only=%d",
                 fold["unique_solvents"], fold["train_unique_solvents"],
                 fold["val_unique_solvents"],
                 len(fold["common_solvents"]), len(fold["val_only_solvents"]))
        log.info("  Redox — train: %s  val: %s",
                 dict(fold["train_redox_counts"]), dict(fold["val_redox_counts"]))


def save_fold_statistics(statistics: list[dict], out_path: Path, meta: dict) -> None:
    """Write solvent distribution statistics to a plain-text file."""
    with open(out_path, "w") as f:
        f.write("SOLVENT DISTRIBUTION STATISTICS\n" + "=" * 80 + "\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        for s in statistics:
            f.write(f"\n{s['name']}:\n")
            f.write(f"  Total samples:   {s['total']}\n")
            f.write(f"  Unique solvents: {s.get('unique_solvents', '?')}\n")
            counts = s.get("solvent_counts") or s.get("train_solvent_counts", Counter())
            f.write("  Solvent counts:\n")
            for smiles, cnt in counts.most_common():
                f.write(f"    {smiles}: {cnt}\n")
            if "common_solvents" in s:
                f.write(f"  Common (train ∩ val): {len(s['common_solvents'])}\n")