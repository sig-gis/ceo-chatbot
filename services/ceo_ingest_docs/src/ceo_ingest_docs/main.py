import logging
import sys
import tempfile
from pathlib import Path

from ceo_ingest_docs.config import load_ingestion_config
from ceo_ingest_docs.settings import IngestionSettings
from ceo_ingest_docs.github_loader import GitHubLoader
from ceo_chatbot_core.storage import GCSStorage


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Load configuration
        config = load_ingestion_config("conf/base/rag_config.yml")
        logging.info(f"Loaded configuration for GitHub repo: {config.repo_url}")

        settings = IngestionSettings()
        bucket_name = settings.docs_bucket_name
        prefix = settings.folder_prefix

        # Initialize components
        loader = GitHubLoader(config)
        gcs = GCSStorage(bucket_name)

        # Perform the extraction and upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logging.info("Cloning repository...")
            cloned_path = loader.clone(temp_path)
            logging.info(f"Repository cloned to: {cloned_path}")

            logging.info("Uploading to Google Cloud Storage...")
            result = gcs.sync_up(cloned_path, prefix)
            print(result)
            logging.info("Upload completed successfully.")

        logging.info("Document extraction pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
