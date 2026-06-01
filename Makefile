.PHONY: install reproduce clean

install:
	pip install -r requirements.txt

reproduce:
	python src/bio_bandage_reproducible_study.py --all

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
