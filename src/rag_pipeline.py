import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.embedding import get_embedding_model, get_or_create_vectorstore

from src.chunking.headerChunking import get_header_chunks


load_dotenv()

def build_rag_chain(retriever):
    tempalte = (
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

    prompt = ChatPromptTemplate.from_template(tempalte)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    embeddings = get_embedding_model()
    DB_path = "./vectorstore/db_faiss"

    if not os.path.exists(DB_path):
        print("Chưa tìm thấy database. Bắt đầu đọc file")
        splits = get_header_chunks(data_path = "./data")
        print(f"Đã chunk xong: Tạo ra {len(splits)} chunks tài liệu")

        vectorstore = get_or_create_vectorstore(
            embeddings = embeddings,
            documents=splits,
            store_path=DB_path
        )
    else:
        vectorstore = get_or_create_vectorstore(embeddings=embeddings, store_path=DB_path)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",   
        search_kwargs={"k": 5}   
    )

    rag_chain = build_rag_chain(retriever)

    print("ChatBot đã sẵn sàng, Gõ 'exit' để thoát")       
    while True:
        user_input = input("Quesion: ").strip()
        if user_input == "exit":
            print("Đang thoát ...")
            break
        print("Đang trả lời...")
        answer = rag_chain.invoke(user_input)
        print(answer)

if __name__ == "__main__":
    main()
    



    

    