.PHONY: setup infra server api edge test benchmark lint clean help

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON     = python3.10
VENV       = .venv
CAM_ID    ?= cam_01
SOURCE    ?= 0

help:
	@echo ""
	@echo "Multi-Camera Defect Tracking — Make Commands"
	@echo "─────────────────────────────────────────────"
	@echo "  make setup       Install dependencies + create config"
	@echo "  make infra        Start Kafka, Postgres, Redis via Docker"
	@echo "  make server       Start central processing server"
	@echo "  make api          Start FastAPI on :8000"
	@echo "  make edge         Start edge node (CAM_ID=cam_01 SOURCE=0)"
	@echo "  make simulate     Start camera simulator (no hardware needed)"
	@echo "  make dashboard    Serve dashboard on :3000"
	@echo "  make test         Run unit tests"
	@echo "  make benchmark    Run performance benchmark"
	@echo "  make lint         Run ruff linter"
	@echo "  make clean        Remove build artifacts"
	@echo ""

setup:
	bash setup.sh

infra:
	docker-compose up -d zookeeper kafka postgres redis
	@echo "▸ Waiting for Kafka to be ready..."
	@sleep 10
	@echo "✓ Infrastructure started"

server:
	$(VENV)/bin/python server/main.py

api:
	$(VENV)/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

edge:
	$(VENV)/bin/python edge/main.py --camera-id $(CAM_ID) --source $(SOURCE)

simulate:
	$(VENV)/bin/python scripts/simulate_cameras.py --cameras 2 --fps 10 --defect-rate 0.3

dashboard:
	$(VENV)/bin/python -m http.server 3000 --directory dashboard
	@echo "Dashboard: http://localhost:3000"

test:
	$(VENV)/bin/pytest tests/server/ tests/edge/ -v --tb=short

test-all:
	$(VENV)/bin/pytest tests/ -v --tb=short

benchmark:
	$(VENV)/bin/python tests/benchmark.py --cameras 4 --duration 30

lint:
	$(VENV)/bin/ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	@echo "✓ Cleaned"

down:
	docker-compose down
