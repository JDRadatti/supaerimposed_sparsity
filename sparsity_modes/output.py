"""
Consistent CSV output for training runs, evaluations, and visualizations.

The ``OutputLogger`` writes all artefacts into a per-run directory::

    {output_dir}/run_{timestamp}_{commit}/
        training_log.csv
        eval_summary.csv
        visualize_<mode>.csv
        visualize_diff_<mode>.csv
"""

from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _get_git_info() -> tuple[str, str]:
    """Return ``(commit_hash, branch_name)``, or ``("unknown", "unknown")``."""
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


class OutputLogger:
    """Writes CSV logs and visualizations into a per-run directory.

    Lifecycle::

        logger = OutputLogger(output_dir="runs/exp", run_name="my_run")
        logger.start_run()
        for epoch in ...:
            logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)
        logger.log_eval_summary(results)
        logger.finish_run()
    """

    def __init__(self, output_dir: str, run_name: str) -> None:
        self._output_dir = output_dir
        self._run_name = run_name
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

    def start_run(self, model_info: dict[str, Any] | None = None) -> str:
        """Create the per-run directory and open ``training_log.csv``.

        Args:
            model_info: Optional dict with keys like ``"model_class"``,
                        ``"total_params"`` to write as metadata.

        Returns:
            The absolute path to the run directory.
        """
        self._commit, self._branch = _get_git_info()
        self._timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        run_dir_name = f"run_{self._timestamp}_{self._commit}"
        self._run_dir = str(Path(self._output_dir) / run_dir_name)
        os.makedirs(self._run_dir, exist_ok=True)

        # Open training CSV
        self._training_csv_path = str(Path(self._run_dir) / "training_log.csv")
        self._training_file = open(self._training_csv_path, "w", newline="")
        self._training_writer = csv.writer(self._training_file, lineterminator="\n")

        # Metadata headers
        self._training_writer.writerow(["# run_name", self._run_name])
        self._training_writer.writerow(["# timestamp", self._timestamp])
        self._training_writer.writerow(["# git_commit", self._commit])
        self._training_writer.writerow(["# git_branch", self._branch])
        if model_info:
            for key, val in model_info.items():
                self._training_writer.writerow([f"# model_{key}", str(val)])

        # Column headers
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
        """Write a single training epoch row to ``training_log.csv``."""
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
        """Write ``eval_summary.csv`` with per-mode evaluation metrics.

        Args:
            results: ``{mode_name: (avg_loss, accuracy)}``.
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

    def open_visualize_csv(
        self,
        mode_name: str,
        header_lines: list[str],
    ) -> tuple[str, Any]:
        """Open a CSV for ``visualize()`` output.

        Header lines are written as ``# ...`` comments.  Data rows should
        be written via ``csv.writer`` on the returned file handle.

        Returns:
            ``(file_path, file_handle)``.
        """
        if self._run_dir is None:
            raise RuntimeError("OutputLogger.start_run() must be called first.")

        path = str(Path(self._run_dir) / f"visualize_{mode_name}.csv")
        f = open(path, "w", newline="")
        for line in header_lines:
            f.write(f"# {line}\n")
        return path, f

    def open_diff_csv(
        self,
        mode_name: str,
        header_lines: list[str],
    ) -> tuple[str, Any]:
        """Open a CSV for ``visualize_diff()`` output.

        Returns:
            ``(file_path, file_handle)``.
        """
        if self._run_dir is None:
            raise RuntimeError("OutputLogger.start_run() must be called first.")

        path = str(Path(self._run_dir) / f"visualize_diff_{mode_name}.csv")
        f = open(path, "w", newline="")
        for line in header_lines:
            f.write(f"# {line}\n")
        return path, f

    def write_overview(self, text: str) -> None:
        """Write ``overview.txt`` with the full run summary.

        Args:
            text: The full text content of the overview file.
        """
        if self._run_dir is None:
            raise RuntimeError("OutputLogger.start_run() must be called first.")

        path = str(Path(self._run_dir) / "overview.txt")
        with open(path, "w") as f:
            f.write(text)
            f.write("\n")

    def finish_run(self) -> None:
        """Close the training CSV file."""
        if self._training_file is not None:
            self._training_file.close()
            self._training_file = None
            self._training_writer = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> str | None:
        """The per-run directory, or ``None`` if ``start_run()`` hasn't been called."""
        return self._run_dir

    @property
    def commit(self) -> str:
        """Short git commit hash (or ``"unknown"``)."""
        return self._commit

    @property
    def branch(self) -> str:
        """Git branch name (or ``"unknown"``)."""
        return self._branch

    @property
    def timestamp(self) -> str:
        """Run timestamp string."""
        return self._timestamp
