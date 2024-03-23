start_chroma:
	docker compose -f docker-compose.dev.yml up -d chromadb

clean:
	docker compose -f docker-compose.dev.yml down --remove-orphans

index_docs:
	python src/index_document.py

main:
	streamlit run src/main.py --server.port 8080

staging_build:
	docker compose -f docker-compose.stage.yml build

staging_up:
	docker compose -f docker-compose.stage.yml build
