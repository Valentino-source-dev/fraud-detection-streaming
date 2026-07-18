.PHONY: help build up down restart logs test lint train clean seed

COMPOSE := docker compose

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Docker ──────────────────────────────────────

build: ## Build all Docker images
	$(COMPOSE) build

up: ## Start the full pipeline (detached)
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d

down: ## Stop all services
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) restart

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

logs-consumer: ## Tail logs for stream-consumer only
	$(COMPOSE) logs -f stream-consumer

logs-generator: ## Tail logs for transaction-generator only
	$(COMPOSE) logs -f transaction-generator

ps: ## Show service status
	$(COMPOSE) ps

# ── Development ─────────────────────────────────

lint: ## Run linting (ruff)
	ruff check generator/ consumer/ training/ tests/
	ruff format --check generator/ consumer/ training/ tests/

format: ## Auto-format code
	ruff format generator/ consumer/ training/ tests/

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --cov=consumer --cov=generator --cov=training --cov-report=term-missing

# ── ML Training ─────────────────────────────────

train: ## Train XGBoost model and log to MLflow
	@echo "Starting MLflow + Postgres for training..."
	$(COMPOSE) up -d postgres mlflow
	@sleep 5
	cd training && python train.py
	@echo "✓ Training complete. Check MLflow at http://localhost:5000"

evaluate: ## Evaluate the latest model
	cd training && python evaluate.py

baseline: ## Train Isolation Forest baseline for comparison
	cd training && python baseline.py

# ── Utilities ───────────────────────────────────

clean: ## Remove all containers, volumes, and generated files
	$(COMPOSE) down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

reset-db: ## Drop and recreate the database
	$(COMPOSE) exec postgres psql -U fraud -d fraud_detection -f /docker-entrypoint-initdb.d/init.sql

urls: ## Show all service URLs
	@echo ""
	@echo "  📊 Grafana:           http://localhost:3000  (admin/admin)"
	@echo "  📦 MLflow:            http://localhost:5000"
	@echo "  📈 Prometheus:        http://localhost:9090"
	@echo "  📨 Redpanda Console:  http://localhost:8080"
	@echo "  📉 Consumer Metrics:  http://localhost:8000/metrics"
	@echo ""
