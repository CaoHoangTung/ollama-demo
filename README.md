# Ollama Demo

## Invoice Extract
### How to run:

1. Install dependencies
- Install pip packages
   ```bash
   pip install -r requirements.txt
   ```
- Install Rust on local
   https://www.rust-lang.org/tools/install

- Install poppler
   https://pdf2image.readthedocs.io/en/latest/installation.html
   
2. Run the main script with the pdf file as the input
   ```bash
   python src/invoice_extract/main.py /path/to/pdf/file --model="llama2"
   ```
   Optionally you can provide a different model.
