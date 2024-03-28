import argparse
import tempfile

from PIL import Image
from pdf2image import convert_from_path

from extract_chain import InvoiceExtractChain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to PDF file")
    parser.add_argument("--model", type=str, default="mistral")
    args = parser.parse_args()
    chain = InvoiceExtractChain(model_name=args.model)

    chain.invoke(args.file)