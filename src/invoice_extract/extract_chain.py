import os
from typing import List

from PIL import Image
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

OLLAMA_URL = os.environ["OLLAMA_URL"] if "OLLAMA_URL" in os.environ else "http://localhost:11434"

system_prompt = """<s>[INST]
You are an excellent document scanner.
I am attaching the content of a multipage-page PDF order document.
I want to extract the line-items of the text which carries any of the phrases "Komplettangebot" or "Komplette PV Anlage",
including the product description and the price as JSON strings.

Also include in the JSON the total order value is, and any tax amounts.

The comma is the decimal separator and the dot is the thousands separator (german numbering).

Extract the name of the customer, the order date and the order number out into JSON as well.
Remove line-breaks and form continuous sentences.

Extract pre-text and post-texts, from before and after the order lines.
Here, leave the line-breaks as they are.
[/INST]</s>
"""

class InvoiceExtractChain:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, model_name: str):
        self.model = ChatOllama(model=model_name, base_url=OLLAMA_URL)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Invoice content: {file_content}. Remember to return result in JSON, and do not provide any explanation"),
        ])

        self.chain = self.chain = (
                self.prompt
                | self.model
                | StrOutputParser()
        )

    def invoke(self, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        pages_content = [page.page_content for page in pages]
        print(self.chain.invoke({"file_content": pages_content}))
