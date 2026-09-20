#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/composectl.sh <build|start|stop|restart|status|logs|pull-model>

Set COMPOSE_FILE=docker-compose.external.yml to use an Ollama instance outside this Compose project.

Commands:
  build       Build the backend and frontend images
  start       Start all Docker Compose services in the background
  stop        Stop all Docker Compose services
  restart     Stop and start all services
  status      Show service status
  logs        Follow logs; optionally pass a service name
  pull-model  Pull OLLAMA_MODEL into the Ollama container
EOF
}

compose() {
  docker compose --project-directory "$ROOT_DIR" -f "${COMPOSE_FILE:-docker-compose.yml}" "$@"
}

case "${1:-}" in
  build)
    compose build
    ;;
  start)
    compose up -d
    ;;
  stop)
    compose down
    ;;
  restart)
    compose down
    compose up -d
    ;;
  status)
    compose ps
    ;;
  logs)
    shift
    compose logs -f "$@"
    ;;
  pull-model)
    if [[ "${COMPOSE_FILE:-docker-compose.yml}" == "docker-compose.external.yml" ]]; then
      echo "External Compose mode does not manage Ollama models. Pull the model from the host Ollama service." >&2
      exit 1
    fi
    model="${OLLAMA_MODEL:-$(awk -F= '/^OLLAMA_MODEL=/{print $2; exit}' .env 2>/dev/null || true)}"
    [[ -n "$model" ]] || { echo "OLLAMA_MODEL is not set in the environment or .env" >&2; exit 1; }
    compose exec ollama ollama pull "$model"
    ;;
  *)
    usage
    exit 1
    ;;
 esac
