import logging
import sys
import tempfile
from pathlib import Path

from ceo_chatbot.config import load_rag_config
from ceo_chatbot.ingest.github_loader import GitHubLoader
from ceo_chatbot.ingest.gcs_uploader import GCSHandler

def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Load configuration
        config = load_rag_config()
        logging.info(f"Loaded configuration for GitHub repo: {config.github_repo_url}")

        # Initialize components
        loader = GitHubLoader(config)
        uploader = GCSHandler(config)

        # Perform the extraction and upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logging.info("Cloning repository...")
            cloned_path = loader.clone(temp_path)
            logging.info(f"Repository cloned to: {cloned_path}")

            logging.info("Uploading to Google Cloud Storage...")
            uploader.upload_docs(cloned_path)
            logging.info("Upload completed successfully.")

        logging.info("Document extraction pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
