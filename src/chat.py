import os

import chromadb
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_HOST = os.environ["CHROMA_HOST"] if "CHROMA_HOST" in os.environ else "localhost"
try:
    CHROMA_PORT = int(os.environ["CHROMA_PORT"]) if "CHROMA_PORT" in os.environ else 8000
except ValueError:
    CHROMA_PORT = 8000

OLLAMA_URL = os.environ["OLLAMA_URL"] if "OLLAMA_URL" in os.environ else "http://localhost:11434"
MODEL_NAME = os.environ["MODEL_NAME"] if "MODEL_NAME" in os.environ else "mistral:instruct"


def get_vector_store():
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return Chroma(client=chroma_client, embedding_function=FastEmbedEmbeddings())


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)
        self.embedding = FastEmbedEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
                <s> [INST] You are an assistant for question-answering tasks. Use only the following pieces of retrieved context 
                to answer the question. If you don't know the answer, just say that you don't know. 
                Provide quotation from the context if needed [/INST] </s> 
                [INST] Question: {question} 
                Context: {context} 
                Answer: [/INST]
                """
        )

    def setup_chain(self):
        self.vector_store = get_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
        )

    def ask(self, query: str):
        # return f"{self.vector_store.search(query, search_type='similarity')}"
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
