from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory containing PDF files
pdf_directory = "data/pdfs"
all_documents = []

# Load all PDF documents
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add metadata to each document
        for doc in documents:
            doc.metadata["source"] = filename

        all_documents.extend(documents)

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(all_documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define Pinecone index name
index_name = "document-embeddings"

# Check if the index exists, if not create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Generate embeddings and upload to Pinecone
for i, split in enumerate(all_splits):
    # Generate embedding for the text chunk
    embedding = embeddings.embed_documents([split.page_content])[0]

    # Prepare metadata
    metadata = {
        "text": split.page_content,
        "source": split.metadata["source"]
    }

    # Upload to Pinecone
    index.upsert([(f"chunk_{i}", embedding, metadata)])

print("Embeddings uploaded to Pinecone successfully!")