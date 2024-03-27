start_chroma:
	docker compose -f docker-compose.dev.yml up -d chromadb

clean_pdf_chat:
	docker compose -f docker-compose.dev.yml down --remove-orphans

pdf_chat:
	streamlit run src/pdf_chat/chat_ui.py --server.port 8080

invoice_extract:
	python src/invoice_extract/main.py

create_invoice_extract_model:
	ollama create "invoice_extract:latest" -f src/invoice_extract/Modelfile

staging_build:
	docker compose -f docker-compose.stage.yml build

staging_up:
	docker compose -f docker-compose.stage.yml build
