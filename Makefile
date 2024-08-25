ENV_FILE=.env
REQUIREMENTS=requirements.txt
VENV=venv

PROJECT_NAME=yase

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install        - Crée un environnement virtuel et installe les dépendances."
	@echo "  tests       - Exécute les tests avec génération de rapports de couverture."
	@echo "  run            - Lance l'application principale."
	@echo "  sonar          - Exécute SonarQube pour analyser le code (si configuré)."
	@echo "  format         - Formate le code avec black."
	@echo "  lint           - Vérifie le code avec flake8."
	@echo "  clean          - Supprime l'environnement virtuel et les fichiers temporaires."

.PHONY: install
install: $(VENV)/bin/activate

$(VENV)/bin/activate: $(REQUIREMENTS)
	python3.9 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r $(REQUIREMENTS)

.PHONY: tests
tests: $(VENV)/bin/activate
	$(VENV)/bin/pytest --cov=src --cov-report=term-missing --cov-report=html tests/ --junitxml=results/pytest-results.xml

.PHONY: run
run: $(VENV)/bin/activate
	$(VENV)/bin/python src/yase/main.py

.PHONY: jupyterlab
jupyterlab: $(VENV)/bin/activate
	$(VENV)/bin/jupyter lab

.PHONY: stop-jupyterlab
stop-jupyterlab:
	@echo "Arrêter JupyterLab..."
	@pkill -f "jupyter-lab"



.PHONY: sonar
sonar: $(VENV)/bin/activate
	sonar-scanner \
		-Dsonar.projectKey=yase \
		-Dsonar.sources=. \
		-Dsonar.host.url=http://localhost:9100 \
		-Dsonar.login=your_sonarqube_token


.PHONY: format
format: $(VENV)/bin/activate
	$(VENV)/bin/black .

.PHONY: lint
lint: $(VENV)/bin/activate
	$(VENV)/bin/flake8 src/ tests/ > results/flake8-report.txt

.PHONY: build
build: $(VENV)/bin/activate
	poetry build
	@echo "Le package a été créé dans le dossier $(DIST_DIR). Pour l'installer, exécutez la commande suivante dans votre environnement virtuel:"
	@echo "pip install $(DIST_DIR)/`ls $(DIST_DIR) | grep .whl`"

.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	find . -type f -name "*.pyo" -exec rm -f {} +
