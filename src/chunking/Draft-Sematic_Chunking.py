import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import  ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
#GoogleGenerativeAIEmbeddings, 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

def rag_chatbot():
    load_dotenv()

    DB_FAISS_PATH = "./vectorstore/db_faiss"

    #Start embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": "cpu"}, # Nếu máy có GPU Nvidia, đổi thành "cuda"
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(DB_FAISS_PATH):
        vectorstore = FAISS.load_local(
            folder_path=DB_FAISS_PATH,
            embeddings=embeddings,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )
    else:
    #Read the document
        loader = DirectoryLoader(
            # Read the file folder data
            path = "./data",
            glob = "**/*.*", #Read all file style
            loader_cls = UnstructuredFileLoader,
            show_progress = True,
            use_multithreading = True
        )

        MARKDOWN_SEPARATORS = [
            #Theo thứ tự từ trên xuống
            "\n#{1,6} ", #Heading 1 -> 6 #Regular Expression
            "```\n", #code block
            "\n\\*\\*\\*+\n",
            "\n---++\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]

        docs = loader.load()
        # print(docs)
        # print(len(docs))

        #Split documents
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size = 1000,
        #     chunk_overlap = 200,
        #     add_start_index = True, #Save the beginning char 1000, 2000, 3000, ...
        #     strip_whitespace = True, #Remove space at begin and the end in a chunk
        #     separators = MARKDOWN_SEPARATORS,
        # )

        #Semantic Chunker - Chậm 
        text_splitter = SemanticChunker(
            embeddings= embeddings,
            breakpoint_threshold_amount=0.85
        )



        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings,
            #Cosine Similarity
            distance_strategy = DistanceStrategy.COSINE
        )

        vectorstore.save_local(DB_FAISS_PATH)


    # retriever = vectorstore.as_retriever(
    #     search_type="similarity_score_threshold", #Chỉ lấy chunk tài liệu có điểm tương đồng của người dùng vượt quá 0.2 
    #     search_kwargs={"k": 5, "score_threshold": 0.2} #Điểm tưởng đồng: 0.2 và chỉ lấy 5 tài liệu có điểm cao nhất
    # )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        # thinking_budget=0  # Tắt chế độ reasoning chuyên sâu nếu muốn trả lời trực tiếp như flash thường
        #temperature=0 #Tính ngẫu nhiên của modal 0 -> 2 . O là ổn đỉnh, 2 là sáng tạo
    )

    #pineplai rag_chain dùng để kết nối các thành phần ở trên lại với nhau
    # "|" là kết nối lại với nhau: ouput của bước trước là input của bước sau

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} #Tạo ra input cho prompt
        | prompt 
        | llm 
        | StrOutputParser() #Trả về câu trả lời của LLM
    )

    while True:
        user_input = input("Quesion: ").strip()
        if user_input == "exit":
            print("Exiting...")
            break
        answer = rag_chain.invoke(user_input)
        print(answer)
if __name__ == '__main__':
    rag_chatbot()
