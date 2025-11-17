from typing import List
import datasets
from tqdm import tqdm
from langchain_core.documents import Document as LangchainDocument


def load_hf_docs(dataset_name: str = "m-ric/huggingface_doc", split: str = "train") -> List[LangchainDocument]:
    """
    Load documents from a Hugging Face dataset and convert to LangChain Documents.
    """
    ds = datasets.load_dataset(dataset_name, split=split)
    docs = [
        LangchainDocument(page_content=row["text"], metadata={"source": row["source"]})
        for row in tqdm(ds, desc=f"Loading dataset {dataset_name}:{split}")
    ]
    return docs
