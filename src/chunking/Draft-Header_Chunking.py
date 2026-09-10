import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def rag_chatbot():
    load_dotenv()

    DB_FAISS_PATH = "./vectorstore/db_faiss"

    # Start embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": "cpu"},  # Nếu máy có GPU Nvidia, đổi thành "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(DB_FAISS_PATH):
        vectorstore = FAISS.load_local(
            folder_path=DB_FAISS_PATH,
            embeddings=embeddings,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True,
        )
    else:
        # Read the document
        loader = DirectoryLoader(
            # Read the file folder data
            path="./data",
            glob="**/*.*",  # Read all file style
            loader_cls=UnstructuredFileLoader,
            show_progress=True,
            use_multithreading=True,
        )

        docs = loader.load()
        # print(docs)
        # print(len(docs))

        # ==========================================
        # Header Chunking
        # ==========================================
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
            ("####", "Header_4"),
            ("#####", "Header_5"),
            ("######", "Header_6"),

        ]

        # 1. Khởi tạo Markdown Header Splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # Giữ lại header trong text để tăng ngữ cảnh khi semantic search
        )

        # 2. Khởi tạo RecursiveCharacterTextSplitter để xử lý các mục quá dài
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            strip_whitespace=True,
        )

        # 3. Tách theo Header và kế thừa metadata gốc (source, file_path,...)
        header_splits = []
        for doc in docs:
            # Tách text của từng file theo header
            sub_splits = markdown_splitter.split_text(doc.page_content)
            for sub_split in sub_splits:
                # Merge metadata từ file gốc (source path) với metadata header mới tách
                merged_metadata = {**doc.metadata, **sub_split.metadata}
                sub_split.metadata = merged_metadata
                header_splits.append(sub_split)

        # 4. Cắt nhỏ tiếp các đoạn vượt quá 1000 tokens
        splits = text_splitter.split_documents(header_splits)

        chunks_data = [
            {
                "chunk_id": i + 1,
                "length": len(chunk.page_content),
                "metadata": chunk.metadata,
                "page_content": chunk.page_content,
            }
            for i, chunk in enumerate(splits)
        ]

        with open("all_chunks_debug.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"Đã lưu chi tiết chunks vào file 'all_chunks_debug.json'")

        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings,
            # Cosine Similarity
            distance_strategy=DistanceStrategy.COSINE,
        )

        vectorstore.save_local(DB_FAISS_PATH)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        '"I don\'t know based on the provided documents."\n'
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
    )

    # pipeline rag_chain dùng để kết nối các thành phần ở trên lại với nhau
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        user_input = input("Question: ").strip()
        if user_input == "exit":
            print("Exiting...")
            break
        answer = rag_chain.invoke(user_input)
        print(answer)


if __name__ == "__main__":
    rag_chatbot()