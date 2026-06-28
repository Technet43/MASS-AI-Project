.PHONY: install test coverage benchmark dashboard docker-build docker-up

install:
	python -m pip install --upgrade pip
	python -m pip install -r shared/requirements.txt
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest shared/tests/ -v

coverage:
	python -m pytest shared/tests/ --cov=shared/core --cov-report=term-missing

benchmark:
	python scripts/benchmark_real_vs_synthetic.py

dashboard:
	python -m streamlit run new_web/dashboard/app.py

docker-build:
	docker build -t mass-ai-dashboard:local .

docker-up:
	docker compose up --build
