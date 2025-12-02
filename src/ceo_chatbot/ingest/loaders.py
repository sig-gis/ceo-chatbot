from typing import List
from langchain_community.document_loaders import rst
from langchain_core.documents import Document as LangchainDocument
import subprocess
from pathlib import Path
import glob

from ceo_chatbot.config import load_document_extraction_config

def load_rst_docs(docs_dir:str="data/ceo-docs") -> List[LangchainDocument]:
    """
    Load all .rst files in `docs_dir` into a list of Langchain Documents
    """
    rst_files = sorted(glob.glob(docs_dir+"/**/*.rst"))
    rst_docs = []
    for rst_file in rst_files:
        loader = rst.UnstructuredRSTLoader(file_path=rst_file, mode="elements")
        rst_doc = loader.load()
        rst_docs.extend(rst_doc)
    return rst_docs

def sync_ceo_docs(output_dir:str= "data/ceo-docs") -> Path:
    """
    Sync local CEO docs data (at `output_dir`) with snapshot of it on the GCS bucket
    """
    
    config = load_document_extraction_config()
    print(f"Syncing CEO docs from gs://{config.gcs_bucket_name}/{config.gcs_prefix}")
    
    if not Path(output_dir).exists():
        print(f"Creating path '{output_dir}'")
        Path(output_dir).mkdir(exist_ok=True)
    
    # Determine the GCS source
    gcs_path = f"gs://{config.gcs_bucket_name}"
    if config.gcs_prefix:
        gcs_path += f"/{config.gcs_prefix}"

    # rsync should allow for version syncing bw local and gcs
    cmd = ["gsutil", "-m","rsync", "-r", gcs_path, str(output_dir)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to upload to GCS: {e.stderr}") from e
    
    return Path(output_dir)


