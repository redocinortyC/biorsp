"""
Download datasets from URLs and store them in a local directory.
"""

import os
import hashlib
import logging
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import yaml

from ..io import load_data

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent / "data/datasets.yml"
with open(REGISTRY_PATH, encoding="utf-8") as f:
    DATASETS = yaml.safe_load(f)

DATA_DIR = Path(os.getenv("SPATIALRSP_DATA_DIR", "data"))


def _make_session():
    """
    Create and configure a requests session with retry strategy.

    This session will log requests and responses for debugging purposes.

    Returns:
        requests.Session: Configured session with retry and logging.
    """
    s = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    def debug_hook(response, *_, **__):
        """
        Log the request and response details for debugging.
        """
        logger.debug(
            "Request: %s %s\n"
            "Headers: %s\n"
            "Body: %s\n"
            "Status: %s %s\n"
            "Headers: %s\n"
            "Body: %s",
            response.request.method,
            response.request.url,
            response.request.headers,
            response.request.body,
            response.status_code,
            response.reason,
            response.headers,
            response.text[:100],
        )

    s.hooks["response"] = [debug_hook]

    return s


SESSION = _make_session()
CHUNK_SIZE = 1024 * 1024


def _verify_checksum(path: Path, sha256: str) -> bool:
    """
    Verify that a file has the specified SHA-256 checksum.

    Args:
        path (Path): The path to the file to verify.
        sha256 (str): The expected SHA-256 checksum.

    Returns:
        bool: True if the file's checksum matches, False otherwise.
    """
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == sha256


def _download(url: str, out: Path, desc: str, sha256: str = None) -> Path:
    """
    Download a file from a URL and save it to disk, with progress bar and optional checksum.

    Args:
        url (str): The URL to download the file from.
        out (Path): The path where the file should be saved.
        desc (str): A description for the download, used in progress bar.
        sha256 (str, optional): The expected SHA-256 checksum of the file.

    Returns:
        Path: The path to the downloaded file.
    """
    logger.debug("Downloading %s to %s", desc, out)
    with SESSION.get(url, stream=True, timeout=10) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(out, "wb") as file, tqdm(
            total=total, unit="iB", unit_scale=True, desc=desc, leave=False
        ) as progress_bar:
            for chunk in resp.iter_content(CHUNK_SIZE):
                file.write(chunk)
                progress_bar.update(len(chunk))

    if sha256 and not _verify_checksum(out, sha256):
        raise IOError(f"Checksum mismatch for {out}")
    logger.info("%s: downloaded → %s", desc, out)
    return out


class DownloadManager:
    """
    Download datasets from URLs and store them in a local directory.
    This class manages the downloading of datasets defined in a YAML registry file.
    It supports multi-threaded downloads and checksum verification.
    Attributes:
        data_dir (Path): The directory where datasets are stored.
        max_workers (int): The maximum number of threads to use for downloading.
        pool (ThreadPoolExecutor): Thread pool for managing download tasks.
    Methods:
        list_datasets() -> List[str]:
            Return a list of available datasets.
        download(name: str, variant: str = None, force: bool = False) -> List[Path]:
            Download a dataset from the registry.

        Args:
            name: The name of the dataset to download.
            variant: The variant of the dataset to download.
            force: Whether to re-download existing files.

        Returns:
            List of Paths to the downloaded (or skipped) files.
    """

    def __init__(self, data_dir: Path = DATA_DIR, max_workers: int = 4):
        """
        Initialize the download manager.

        Args:
            data_dir: The directory where datasets are stored.
            max_workers: The maximum number of threads to use for downloading.
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers, thread_name_prefix="dler")
        self.max_workers = max_workers

    def list_datasets(self) -> List[str]:
        """
        Return a list of available datasets.
        """
        logger.debug("Listing available datasets")
        return list(DATASETS.keys())

    def download(
        self, name: str, variant: str = None, force: bool = False
    ) -> List[Path]:
        """
        Download a dataset from the registry.

        Args:
            name: The name of the dataset to download.
            variant: The variant of the dataset to download.
            force: Whether to re-download existing files.

        Returns:
            List of Paths to the downloaded (or skipped) files.
        """
        logger.debug("Downloading dataset %s", name)
        if name not in DATASETS:
            raise KeyError(f"Dataset '{name}' not in registry")

        raw = DATASETS[name]
        selected = raw.get(variant, raw) if variant else raw
        entries = selected if isinstance(selected, list) else [selected]

        tasks = []
        skipped = []

        for meta in entries:
            url = meta["url"]
            fname = meta["filename"]
            sha256 = meta.get("sha256")
            dest = self.data_dir / fname
            desc = f"{name}/{variant or 'default'}"

            if not force and dest.exists():
                if sha256 is None or _verify_checksum(dest, sha256):
                    logger.info(
                        "%s: already present, skipping download → %s", desc, dest
                    )
                    skipped.append(dest)
                    continue

            tasks.append((url, dest, desc, force, sha256))
            logger.debug("Downloading %s to %s", desc, dest)

        if not tasks:
            logger.debug("All files already exist → returning skipped list")
            return skipped

        def runner(args):
            url, dest, desc, _, checksum = args
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                return _download(url, dest, desc, checksum)
            except Exception as e:
                logger.error("%s: failed → %s", desc, e)
                raise

        results = list(self.pool.map(runner, tasks))
        logger.debug("Downloaded %d new files", len(results))
        return skipped + results


_dm = DownloadManager()
"""The global download manager instance."""


def download_hcl(**kw):
    """Download the HCL dataset."""
    return _dm.download("hcl", **kw)[0]


def download_kpmp(variant="sn", **kw):
    """Download the KPMP dataset with the specified variant."""
    return _dm.download("kpmp", variant, **kw)[0]


def fetch_adata(name, variant=None, **load_kwargs):
    """
    Download a dataset and load it into an AnnData object.

    Args:
        name: The name of the dataset to download.
        variant: The variant of the dataset to download.
        **load_kwargs: Keyword arguments passed to `load_data`.

    Returns:
        The downloaded dataset as an AnnData object.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if variant is not None and not isinstance(variant, str):
        raise TypeError("variant must be a string or None")
    if not isinstance(load_kwargs, dict):
        raise TypeError("load_kwargs must be a dictionary")

    paths = _dm.download(name, variant)
    h5ad = next((p for p in paths if p.suffix == ".h5ad"), paths[0])

    return load_data(h5ad, **load_kwargs)
