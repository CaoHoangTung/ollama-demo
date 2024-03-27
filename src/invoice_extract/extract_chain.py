import base64
import os
from io import BytesIO

import ollama
from PIL import Image

OLLAMA_URL = os.environ["OLLAMA_URL"] if "OLLAMA_URL" in os.environ else "http://localhost:11434"

system_prompt = """You are an agent that will read a list of images from one invoice and extract information from them.
Return the following info:
- customer_name: The name of the customer
- customer_id: The ID of the customer
- order_total: The total amount of the order
"""


def pil_image_to_bytes(image: Image.Image) -> bytes:
    return image.tobytes("raw", "RGB")


class InvoiceExtractChain:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, model_name: str):
        self.model = ollama.Client(host=OLLAMA_URL)
        self.model_name = model_name

    def invoke(self, files: Image):
        image_byte = [pil_image_to_bytes(file) for file in files]
        print("Invoking input on Ollama model", self.model_name)
        return self.model.chat(
            self.model_name,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please parse this invoice", "images": image_byte}
            ],
        )
