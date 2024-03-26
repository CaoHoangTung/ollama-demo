import os
from datetime import datetime
from pathlib import Path

import streamlit
import validators
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chat_agent import get_vector_store


class PDFIndexer:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=50)
        self.vector_store = get_vector_store()
        self.upload_folder = ".data/uploads"
        os.makedirs(self.upload_folder, exist_ok=True)

    @streamlit.cache_data()
    def list_files(_self):
        files = [
            {
                "name": f.name,
                "uploaded_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for f in Path(_self.upload_folder).iterdir() if f.is_file()
        ]
        return files

    def upload_file(self, st_file):
        if (st_file is None) or (st_file.name is None) or (st_file.type != "application/pdf"):
            raise Exception("Please select a valid PDF file")
        file_bytes = st_file.getbuffer()
        file_path = Path(self.upload_folder, st_file.name)
        if file_path.is_file():
            raise Exception("File is already uploaded")
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        self._index_file(file_path)
        self.list_files.clear()

    def _index_file(self, file_path: Path):
        docs = PyPDFLoader(str(file_path)).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        self.vector_store.add_documents(chunks)
