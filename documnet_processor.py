from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_chroma import Chroma

import os

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')
        
    def load_documents(self, directory: str) -> list[Document]: 
        """Load documents from different file types"""
        loaders = {
            '.pdf': DirectoryLoader(path=directory, glob='**/*.pdf', loader_cls=PyPDFLoader),
            '.txt': DirectoryLoader(path=directory, globe='**/*.txt', loader_cls=TextLoader),
            '.md': DirectoryLoader(path=directory, glob='**/*.md', loader_cls=UnstructuredMarkdownLoader)
        }
        
        documents = []
        for file_type, loader in loaders.items():
            try:
                documents.extend(loader.load())
                print(f'Loaded {file_type} documents')
            except Exception as e:
                print(f'Error loading {file_type} documents: {str(e)}')               
        
        return documents
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: list[Document], persist_directory: str) -> Chroma:
        """Create and persist vector store if it doesn't exist, otherwise load existing one"""
        # Check if persist_directory exists & has content
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f'Loading existing vector store from {persist_directory}')
            # Load existing vector store
            vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )             
            return vector_store
        
        print(f'Creating new vector store in {persist_directory}')
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create new vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return vector_store
        
        