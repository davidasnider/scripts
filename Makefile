.PHONY: check-tesseract install-tesseract install

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