import tempfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
from ..config import DocumentExtractionConfig


class GitHubLoader:
    """
    Handles cloning GitHub repositories to temporary directories for document extraction.
    """

    def __init__(self, config: DocumentExtractionConfig):
        self.config = config

    def clone(self, temp_dir: Optional[Path] = None) -> Path:
        """
        Clone the specified GitHub repository to a temporary directory.

        Args:
            temp_dir: Optional temporary directory to use. If None, creates one.

        Returns:
            Path to the cloned repository (or specified sub-path).

        Raises:
            subprocess.CalledProcessError: If git clone fails.
        """
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())

        repo_name = urlparse(self.config.github_repo_url).path.strip('/').split('/')[-1]

        clone_path = temp_dir / repo_name

        # Git clone command
        cmd = ["git", "clone", "--depth", "1", "--branch", self.config.github_ref, self.config.github_repo_url, str(clone_path)]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e.stderr}") from e

        # If a sub-path is specified, return that path
        if self.config.github_path:
            result_path = clone_path / self.config.github_path
            if not result_path.exists():
                raise FileNotFoundError(f"Specified path '{self.config.github_path}' not found in cloned repository.")
            return result_path
        else:
            return clone_path
