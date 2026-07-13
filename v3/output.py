"""
Consistent CSV output for training runs, evaluation, and visualizations.

The `OutputLogger` is a self‑contained module (also usable by the gate)
that writes all artefacts into a per‑run directory:

    {output_dir}/{filename_prefix}/run_{timestamp}_{commit}/
        training_log.csv
        eval_summary.csv
        checkpoint.pt
"""

from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _get_git_info() -> tuple[str, str]:
    """Return `(commit_hash, branch_name)`, or `("unknown", "unknown")`."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        branch = "unknown"
    return commit, branch


# Project root is the parent of the v3/ package directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_output_dir(path: str) -> str:
    """If *path* is relative, anchor it to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


class OutputLogger:
    """Writes CSV logs and checkpoints into a per‑run directory.

    Lifecycle::

        logger = OutputLogger(OutputOptions(output_dir="./runs", prefix="exp"))
        run_dir = logger.start_run("my_experiment", profiles_info, model_info)
        for epoch in ...:
            logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)
        logger.save_checkpoint(state_dict, ...)
        logger.finish_run()

    Parameters
    ----------
    output_options: `OutputOptions` controlling where files are written.
    """

    def __init__(self, output_options: Any) -> None:
        self._opts = output_options
        self._run_dir: str | None = None
        self._training_csv_path: str | None = None
        self._training_writer: Any = None
        self._training_file: Any = None
        self._commit: str = "unknown"
        self._branch: str = "unknown"
        self._timestamp: str = ""

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str,
        profiles_info: dict[str, Any],
        model_info: dict[str, Any],
    ) -> str:
        """Create the per‑run directory and open `training_log.csv` for writing.

        Args:
            run_name:           Human‑readable name for the run (written into CSV header).
            profiles_info:      Dict with keys `"mode_profiles"`, `"freeze_profiles"`,
                                `"training_profiles"` listing profile names and their
                                parameters.
            model_info:         Dict with keys like `"model_class"`, `"total_params"`,
                                etc.

        Returns:
            The absolute path to the run directory.
        """
        self._commit, self._branch = _get_git_info()
        self._timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Build directory: output_dir/prefix/run_<ts>_<commit>/
        prefix_dir = Path(_resolve_output_dir(self._opts.output_dir)) / self._opts.filename_prefix
        run_dir_name = f"run_{self._timestamp}_{self._commit}"
        self._run_dir = str(prefix_dir / run_dir_name)
        os.makedirs(self._run_dir, exist_ok=True)

        # Open training CSV for streaming writes.
        self._training_csv_path = str(Path(self._run_dir) / "training_log.csv")
        self._training_file = open(self._training_csv_path, "w", newline="")
        self._training_writer = csv.writer(self._training_file, lineterminator="\n")

        # Write header metadata.
        self._training_writer.writerow(["# run_name", run_name])
        self._training_writer.writerow(["# timestamp", self._timestamp])
        self._training_writer.writerow(["# git_commit", self._commit])
        self._training_writer.writerow(["# git_branch", self._branch])
        for key, val in model_info.items():
            self._training_writer.writerow([f"# model_{key}", str(val)])

        # Profile info.
        for cat, entries in profiles_info.items():
            if entries:
                self._training_writer.writerow(
                    [f"# {cat}", ", ".join(str(e) for e in entries)]
                )

        # Column headers.
        self._training_writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        self._training_file.flush()

        return self._run_dir

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Write a single training epoch row to `training_log.csv`."""
        if self._training_writer is None:
            raise RuntimeError("OutputLogger.start_run() must be called first.")
        self._training_writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.4f}",
                f"{val_loss:.6f}",
                f"{val_acc:.4f}",
                f"{lr:.8f}",
            ]
        )
        self._training_file.flush()

    def log_eval_summary(self, results: dict[str, tuple[float, float]]) -> None:
        """Write `eval_summary.csv` with per‑mode evaluation metrics.

        Args:
            results:  `{mode_name: (avg_loss, accuracy)}`.
        """
        if self._run_dir is None:
            raise RuntimeError("OutputLogger.start_run() must be called first.")

        path = str(Path(self._run_dir) / "eval_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["# timestamp", self._timestamp])
            w.writerow(["# git_commit", self._commit])
            w.writerow(["mode", "loss", "accuracy"])
            for mode, (loss, acc) in results.items():
                w.writerow([mode, f"{loss:.6f}", f"{acc:.4f}"])

    def finish_run(self) -> None:
        """Close the training CSV file."""
        if self._training_file is not None:
            self._training_file.close()
            self._training_file = None
            self._training_writer = None

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def open_visualize_csv(
        self,
        mode_name: str,
        header_lines: list[str],
    ) -> tuple[str, Any]:
        """Open a CSV file for `visualize()` output and return its path.

        Header lines are written directly (no csv quoting) so they act as
        human‑readable comments.  Data rows should be written via
        `csv.writer` on the returned file handle.

        Args:
            mode_name:      Name of the mode being visualized.
            header_lines:   Lines to write as `# ...` comments.

        Returns:
            `(file_path, file_handle)`.
        """
        if self._run_dir is None:
            raise RuntimeError("OutputLogger.start_run() must be called first.")

        path = str(Path(self._run_dir) / f"visualize_{mode_name}.csv")
        f = open(path, "w", newline="")
        for line in header_lines:
            f.write(f"# {line}\n")
        return path, f

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> str | None:
        """The per‑run directory, if `start_run()` has been called."""
        return self._run_dir

    @property
    def commit(self) -> str:
        """Short git commit hash (or `"unknown"`)."""
        return self._commit

    @property
    def branch(self) -> str:
        """Git branch name (or `"unknown"`)."""
        return self._branch
