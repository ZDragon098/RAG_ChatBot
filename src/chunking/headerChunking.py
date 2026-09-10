import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def get_header_chunks(data_path: str = "./data", debug_file: str = "all_chunks_debug.json"):
    loader = PyPDFDirectoryLoader(data_path)
    docs = loader.load()

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
        ("#####", "Header_5"),
        ("######", "Header_6"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
    )

    header_splits = []
    for doc in docs:
        sub_splits = markdown_splitter.split_text(doc.page_content)
        for sub_split in sub_splits:
            merged_metadata = {**doc.metadata, **sub_split.metadata}
            sub_split.metadata = merged_metadata
            header_splits.append(sub_split)
    
    splits = text_splitter.split_documents(header_splits)

    if(debug_file):
        chunks_data = [
            {
                "chunk_id": i + 1,
                "length": len(chunk.page_content),
                "metadata": chunk.metadata,
                "page_content": chunk.page_content,
            }
            for i, chunk in enumerate(splits)
        ]

        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"Da luu chi tiet chunks vao file '{debug_file}'")
    
    return splits