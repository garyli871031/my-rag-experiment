import hashlib
from pathlib import Path
from typing import Iterator, Set

# Extend this set with whatever file types your unstructured/LangChain parser handles
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".png", ".jpg", ".jpeg"}


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of a file to prevent duplicate ingestion."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_local_files(
    root_dir: str | Path = "./docs",
    extensions: Set[str] = SUPPORTED_EXTENSIONS,
) -> Iterator[dict]:
    """
    Recursively yields file paths and metadata from a local folder.
    Uses a generator so large folders don't spike RAM.
    """
    base_path = Path(root_dir).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    for path in sorted(base_path.rglob("*")):
        # Skip directories, hidden files (e.g. .DS_Store), and unsupported formats
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in extensions:
            continue

        yield {
            "path": str(path),
            "filename": path.name,
            "relative_path": str(path.relative_to(base_path)),
            "extension": path.suffix.lower(),
            "file_hash": compute_file_hash(path),
        }