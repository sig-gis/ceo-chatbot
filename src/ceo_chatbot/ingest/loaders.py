from typing import List
from langchain_community.document_loaders import rst
from langchain_core.documents import Document as LangchainDocument
import subprocess
from pathlib import Path
import glob

from ceo_chatbot.config import load_document_extraction_config

def load_rst_docs(docs_dir: str = "data/ceo-docs") -> List[LangchainDocument]:
    """
    Load all .rst files in `docs_dir` into a list of Langchain Documents.

    details:
        Uses UnstructuredRSTLoader to parse RST files into elements.
        Calculates and adds a ReadTheDocs URL to each document's metadata
        based on its file path within the documentation structure.
    """
    base_url = "https://collect-earth-online-doc.readthedocs.io/en/latest/"
    docs_path = Path(docs_dir)
    rst_files = sorted(glob.glob(str(docs_path / "**/*.rst"), recursive=True))
    
    rst_docs = []
    for rst_file_str in rst_files:
        rst_file = Path(rst_file_str)
        
        # Calculate the relative path for the URL
        try:
            # Try to get path relative to 'source' if it exists in the path
            if "source" in rst_file.parts:
                source_idx = rst_file.parts.index("source")
                rel_parts = rst_file.parts[source_idx + 1:]
            else:
                rel_parts = rst_file.relative_to(docs_path).parts
            
            # Convert .rst to .html for the URL
            rel_url_path = "/".join(rel_parts).replace(".rst", ".html")
            doc_url = f"{base_url}{rel_url_path}"
        except Exception:
            doc_url = base_url

        loader = rst.UnstructuredRSTLoader(file_path=str(rst_file), mode="elements")
        elements = loader.load()
        
        # Add URL to metadata of each element
        for doc in elements:
            doc.metadata["url"] = doc_url
            
        rst_docs.extend(elements)
        
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


