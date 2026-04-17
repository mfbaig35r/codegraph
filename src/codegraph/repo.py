"""Repo acquisition: local path validation and git clone."""

import hashlib
import subprocess
from pathlib import Path


def _is_git_url(source: str) -> bool:
    """Return True if source looks like a git URL."""
    return (
        source.startswith("https://")
        or source.startswith("git@")
        or source.startswith("http://")
        or source.endswith(".git")
    )


def _repo_hash(source: str) -> str:
    """Generate a short hash for a repo source."""
    normalized = str(Path(source).resolve()) if not _is_git_url(source) else source
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def _clone_repo(url: str, target_dir: Path) -> Path:
    """Clone a git repo into target_dir and return the path."""
    repo_hash = _repo_hash(url)
    clone_path = target_dir / repo_hash
    if clone_path.exists():
        # Already cloned — pull latest
        subprocess.run(
            ["git", "-C", str(clone_path), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return clone_path
    clone_path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, str(clone_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return clone_path


def resolve_repo(source: str, clone_dir: Path) -> tuple[str, str]:
    """
    Resolve a repo source to (repo_id, absolute_repo_path).

    source: local directory path or git clone URL.
    clone_dir: directory to clone into (for URLs).
    """
    if _is_git_url(source):
        clone_dir.mkdir(parents=True, exist_ok=True)
        repo_path = _clone_repo(source, clone_dir)
        return _repo_hash(source), str(repo_path)

    path = Path(source).resolve()
    if not path.exists():
        raise ValueError(f"Path does not exist: {source}")
    if not path.is_dir():
        raise ValueError(f"Not a directory: {source}")
    if path.is_symlink():
        raise ValueError(f"Symlinks not supported: {source}")
    # Reject system directories
    for forbidden in ("/etc", "/var", "/usr", "/bin", "/sbin", "/sys", "/proc"):
        if str(path).startswith(forbidden):
            raise ValueError(f"Cannot index system directory: {path}")
    return _repo_hash(source), str(path)
