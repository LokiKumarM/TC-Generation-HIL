from doc_loader import load_csv, load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import os

csv_file_path = "Req EPS/SystemReq_EPS.csv"
pdf_file_path = "Req EPS/EPS_Info.pdf"

requirements_csv = load_csv(csv_file_path)
req_instance = requirements_csv[14]['Primary Text']

requirements_pdf = load_pdf(pdf_file_path)

def for_embedding():
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really chunk size, just to show.
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    requirement_texts = text_splitter.create_documents([requirements_pdf])

    #writing metadata manaully

    for items in range(0, len(requirement_texts)):
        requirement_texts[items].metadata = {'source': 'EPS_Info.pdf'}

    #Loading csv data for creating knowledge base

    loader = CSVLoader(file_path=csv_file_path)
    final_documents = loader.load()

    for item in requirement_texts:
        final_documents.append(item)

    return final_documents

def create_vectorstore():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore_path = "chromadb_store_new"
    unique_collection_name = f"collection_{uuid.uuid4()}"

    if os.path.exists(vectorstore_path):
        # Load existing Chroma vector store
        print("Loading existing Chroma vector store...")
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings_model)

    else:
        print("Creating new Chroma vector store...")
        requirements_docs = for_embedding()

        persistent_client = chromadb.PersistentClient(vectorstore_path)

        ids = [str(uuid.uuid4()) for _ in range(len(requirements_docs))]

        # collection.add(documents=requirement_texts, ids=ids)

        # collection = persistent_client.get_or_create_collection(unique_collection_name)

        vectorstore = Chroma(
            client=persistent_client,
            collection_name=unique_collection_name,
            embedding_function=embeddings_model,
        )

        vectorstore.add_documents(documents=requirements_docs, ids=ids)

    return vectorstore