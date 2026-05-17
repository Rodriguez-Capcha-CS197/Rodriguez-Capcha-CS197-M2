"""Helpers for locating BEIR-format datasets used by batch jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def is_beir_dataset_dir(path: Path) -> bool:
    """Return True when path contains the standard BEIR dataset files."""
    qrels_dir = path / "qrels"
    return (
        path.is_dir()
        and (path / "corpus.jsonl").is_file()
        and (path / "queries.jsonl").is_file()
        and qrels_dir.is_dir()
        and any(qrels_dir.glob("*.tsv"))
    )


def candidate_beir_dataset_dirs(base_path: Path, dataset_name: str) -> list[Path]:
    """Return likely dataset locations under a user-provided base path."""
    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    add(base_path)
    if base_path.name != dataset_name:
        add(base_path / dataset_name)

    if base_path.is_dir():
        for child in sorted(base_path.iterdir()):
            if child.is_dir():
                add(child)
        for child in sorted(base_path.glob(f"*/{dataset_name}")):
            if child.is_dir():
                add(child)

    return candidates


def resolve_beir_dataset_path(
    data_dir: str,
    dataset_name: str,
    split: str,
    *,
    allow_download: bool = False,
    downloader: Callable[[str], str] | None = None,
    auto_download_root: str = "outputs/beir_data",
) -> str:
    """Resolve a BEIR dataset directory and validate required files for the split."""
    base_path = Path(data_dir)
    candidates = candidate_beir_dataset_dirs(base_path, dataset_name)

    for candidate in candidates:
        if is_beir_dataset_dir(candidate):
            _validate_beir_split(candidate, split, dataset_name)
            return str(candidate)

    should_attempt_download = downloader is not None and (allow_download or data_dir == auto_download_root)
    if should_attempt_download:
        downloaded_path = Path(downloader(str(base_path)))
        for candidate in candidate_beir_dataset_dirs(downloaded_path, dataset_name):
            if is_beir_dataset_dir(candidate):
                _validate_beir_split(candidate, split, dataset_name)
                return str(candidate)
        raise FileNotFoundError(
            f"Downloaded {dataset_name} under '{downloaded_path}', but the BEIR files are still missing. "
            f"Expected corpus.jsonl, queries.jsonl, and qrels/{split}.tsv."
        )

    checked_paths = ", ".join(f"'{candidate}'" for candidate in candidates)
    raise FileNotFoundError(
        f"{dataset_name} dataset not found. Checked these locations for BEIR files "
        f"(corpus.jsonl / queries.jsonl / qrels/{split}.tsv): {checked_paths}. "
        f"{_download_hint(dataset_name, downloader is not None)}"
    )


def _validate_beir_split(path: Path, split: str, dataset_name: str) -> None:
    qrels_path = path / "qrels" / f"{split}.tsv"
    if not qrels_path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} dataset found at '{path}', but missing split file '{qrels_path.name}'. "
            f"Expected corpus.jsonl, queries.jsonl, and qrels/{split}.tsv."
        )


def _download_hint(dataset_name: str, has_downloader: bool) -> str:
    if has_downloader:
        return (
            f"Re-run with --download, or point --data-dir at an existing {dataset_name} "
            "BEIR dataset directory."
        )
    return (
        f"Point the job at an existing {dataset_name} BEIR dataset directory. "
        "This repo does not download that dataset automatically."
    )
