# Définir le nom de l'environnement virtuel
VENV_NAME = venv

# Définir le nom de l'image Docker
DOCKER_IMAGE = maramaram/maram_bouaziz_4ds5_mlops:latest

# Installer les dépendances
install:
	@echo "Installation des dépendances..."
	@pip install -r requirements.txt

# Créer un environnement virtuel
venv:
	@echo "Création de l'environnement virtuel..."
	@python3 -m venv $(VENV_NAME)

# Activer l'environnement virtuel
activate:
	@echo "Activez votre environnement virtuel avec la commande suivante :"
	@echo "source $(VENV_NAME)/bin/activate"

# Vérification du code
lint:
	@echo "Vérification du code avec flake8..."
	@flake8 --max-line-length=120

# Préparer les données
prepare:
	@echo "Préparation des données..."
	@python3 main.py --prepare

# Entraîner le modèle
train:
	@echo "Entraînement du modèle..."
	@python3 main.py --train

# Évaluer le modèle
evaluate:
	@echo "Évaluation du modèle..."
	@python3 main.py --evaluate

# Exécuter les tests
test:
	@echo "Exécution des tests..."
	@pytest --maxfail=3 --disable-warnings -q

# Nettoyage des fichiers inutiles
clean:
	@echo "Nettoyage des fichiers inutiles..."
	@rm -rf __pycache__ *.pyc $(VENV_NAME)

# Commande par défaut qui prépare l'environnement et entraîne le modèle
all: install train

# Construire l'image Docker
docker-build:
	@echo "Construction de l'image Docker..."
	@docker build -t $(DOCKER_IMAGE) .

# Pousser l'image Docker sur Docker Hub
docker-push: docker-build
	@echo "Poussée de l'image Docker sur Docker Hub..."
	@docker push $(DOCKER_IMAGE)

# Exécuter le conteneur Docker localement
docker-run:
	@echo "Exécution du conteneur Docker localement..."
	@docker run -p 5000:5000 $(DOCKER_IMAGE)

# Nettoyer les images Docker non utilisées
docker-clean:
	@echo "Nettoyage des images Docker non utilisées..."
	@docker image prune -f

# Pour afficher les tâches disponibles
help:
	@echo "Tâches disponibles:"
	@echo "  install         - Installer les dépendances"
	@echo "  venv            - Créer un environnement virtuel"
	@echo "  activate        - Afficher la commande pour activer l'environnement virtuel"
	@echo "  lint            - Vérification du code avec flake8"
	@echo "  prepare         - Préparer les données"
	@echo "  train           - Entraîner le modèle"
	@echo "  evaluate        - Évaluer le modèle"
	@echo "  test            - Exécuter les tests"
	@echo "  clean           - Nettoyer les fichiers inutiles"
	@echo "  all             - Installer et entraîner le modèle (par défaut)"
	@echo "  docker-build    - Construire l'image Docker"
	@echo "  docker-push     - Pousser l'image Docker sur Docker Hub"
	@echo "  docker-run      - Exécuter le conteneur Docker localement"
	@echo "  docker-clean    - Nettoyer les images Docker non utilisées"
