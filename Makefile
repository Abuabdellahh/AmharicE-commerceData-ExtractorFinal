# Python virtual environment
ENV_DIR = venv

# Python interpreter
PYTHON = $(ENV_DIR)/Scripts/python

# Install dependencies
install:
	@echo "Creating virtual environment..."
	python -m venv $(ENV_DIR)
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/

# Run linting
lint:
	@echo "Running linting..."
	$(PYTHON) -m flake8 src/

# Format code
format:
	@echo "Formatting code..."
	$(PYTHON) -m black src/

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(ENV_DIR)
	rm -rf __pycache__
	rm -rf src/**/*.pyc
	rm -rf src/**/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

# Run data collection
collect_data:
	@echo "Collecting data from Telegram..."
	$(PYTHON) src/data/telegram_scraper.py

# Run preprocessing
preprocess:
	@echo "Preprocessing data..."
	$(PYTHON) src/data/text_preprocessor.py

# Run NER labeling
label:
	@echo "Labeling data..."
	$(PYTHON) src/labeling/conll_labeler.py

# Train model
train:
	@echo "Training model..."
	$(PYTHON) src/models/ner_model.py

# Compare models
compare:
	@echo "Comparing models..."
	$(PYTHON) src/models/model_comparison.py

# Run vendor analytics
analytics:
	@echo "Running vendor analytics..."
	$(PYTHON) src/analytics/vendor_analytics.py

.PHONY: install test lint format clean collect_data preprocess label train compare analytics
