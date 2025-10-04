.PHONY: check-tesseract install-tesseract install start-ollama pull-models setup-ollama

check-tesseract:
	@if command -v tesseract > /dev/null 2>&1; then \
		echo "Tesseract is already installed."; \
	else \
		echo "Tesseract not found. Installing..."; \
		$(MAKE) install-tesseract; \
	fi

install-tesseract:
	brew install tesseract tesseract-lang

install:
	uv sync

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

setup-ollama: start-ollama pull-models