from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os


pdf_directory = "pdfs"
all_documents = []

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = filename

        all_documents.extend(documents) 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(all_documents)

_ = vector_store.add_documents(documents=all_splits)