.PHONY: check-tesseract install-tesseract check-ollama install-ollama install start-ollama pull-models setup-ollama run-models

check-tesseract:
	@if command -v tesseract > /dev/null 2>&1; then \
		echo "Tesseract is already installed."; \
	else \
		echo "Tesseract not found. Installing..."; \
		$(MAKE) install-tesseract; \
	fi

install-tesseract:
	brew install tesseract tesseract-lang

check-ollama:
	@if command -v ollama > /dev/null 2>&1; then \
		echo "Ollama is already installed."; \
	else \
		echo "Ollama not found. Installing..."; \
		$(MAKE) install-ollama; \
	fi

install-ollama:
	brew install ollama

install:
	uv sync
	$(MAKE) check-tesseract
	$(MAKE) check-ollama
	$(MAKE) setup-ollama

start-ollama:
	@if pgrep -x "ollama" > /dev/null; then \
		echo "Ollama is already running."; \
	else \
		echo "Starting Ollama..."; \
		ollama serve & \
	fi

pull-models:
	ollama pull llama3:70b-instruct
	ollama pull deepseek-coder-v2:16b-lite-instruct
	ollama pull llava:34b-v1.6
	ollama pull nomic-embed-text

setup-ollama: start-ollama pull-models

run-models:
	$(MAKE) start-ollama
	@echo "Ensuring all downloaded models are available..."
	@sleep 2
	@echo "Available models:"
	@ollama list
	@echo "All downloaded Ollama models are ready to use!"

run:
	uv run streamlit run app.py

clean:
	rm -f data/manifest.json
	rm -rf data/chromadb
	rm -rf data/frames