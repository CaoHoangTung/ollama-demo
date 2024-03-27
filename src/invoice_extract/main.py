import argparse
import tempfile

from PIL import Image
from pdf2image import convert_from_path

from extract_chain import InvoiceExtractChain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to PDF file")
    parser.add_argument("--model", type=str, default="llava")
    args = parser.parse_args()
    chain = InvoiceExtractChain(model_name=args.model)

    with tempfile.TemporaryDirectory() as output_folder:
        output_folder = "./.data/extract"
        images = convert_from_path(args.file, output_folder=output_folder, dpi=100, fmt="jpeg")
        print(
            f"Converted PDF into {len(images)} images. Invoking LLM with those images. The sizes of the image is: {images[0].size}")
        result = chain.invoke(images)
        print(result)
