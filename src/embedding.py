import os
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy 
from langchain_huggingface import HuggingFaceEmbeddings 

DB_FAISS_PATH = "./vectorstore/db_faiss"

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name = "bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )    
    return embeddings

def get_or_create_vectorstore(documents = None, embeddings = None, store_path: str = "./vectorstore/db_faiss"):
    if embeddings is None:
        embeddings = get_embedding_model()
    
    if os.path.exists(store_path):
        print(f"Đang tải vectorstore từ '{store_path}'")
        vectorstore = FAISS.load_local(
            store_path,
            embeddings,
            distance_strategy = DistanceStrategy.COSINE,
            allow_dangerous_deserialization = True
        )
        return vectorstore
    
    if documents is None:
        print("Vui lòng cung cấp đường dẫn tài liệu cần trỏ dữ liệu.")
        raise FileNotFoundError("Không tìm thấy tài liệu nguồn để nhúng dữ liệu.")
    
    print(f"Đang tạo vectorstore mới và lưu vào '{store_path}'")

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        distance_strategy = DistanceStrategy.COSINE
    )
    vectorstore.save_local(store_path)
    return vectorstore
    
    
    