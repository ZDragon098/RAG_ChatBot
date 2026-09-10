from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker


def get_sematic_chunks(data_path: str = "./data", embeddings = None):
    loader = PyPDFDirectoryLoader(data_path)
    docs = loader.load()

    text_splitter = SemanticChunker(
        embeddings = embeddings,
        breakpoint_threshold_amount=0.85
    )
    splits = text_splitter.split_documents(docs)
    return splits
    