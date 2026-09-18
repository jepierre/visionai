#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/output/run"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"
FRONTEND_PID_FILE="$RUNTIME_DIR/frontend.pid"
BACKEND_LOG_FILE="$RUNTIME_DIR/backend.log"
FRONTEND_LOG_FILE="$RUNTIME_DIR/frontend.log"

mkdir -p "$RUNTIME_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/appctl.sh <start|stop|restart|status>

Commands:
  start    Start backend and frontend in the background
  stop     Stop backend and frontend started by this script
  restart  Stop both apps, then start them again
  status   Show whether each app is currently running
EOF
}

is_running() {
  local pid_file="$1"

  [[ -f "$pid_file" ]] || return 1

  local pid
  pid="$(<"$pid_file")"
  [[ -n "$pid" ]] || return 1

  kill -0 "$pid" 2>/dev/null
}

start_backend() {
  if is_running "$BACKEND_PID_FILE"; then
    echo "Backend already running (pid $(<"$BACKEND_PID_FILE"))."
    return
  fi

  nohup bash -lc "cd '$ROOT_DIR' && source .venv/bin/activate && exec uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000" \
    >"$BACKEND_LOG_FILE" 2>&1 &
  echo $! >"$BACKEND_PID_FILE"
  echo "Started backend (pid $(<"$BACKEND_PID_FILE")). Log: $BACKEND_LOG_FILE"
}

start_frontend() {
  if is_running "$FRONTEND_PID_FILE"; then
    echo "Frontend already running (pid $(<"$FRONTEND_PID_FILE"))."
    return
  fi

  nohup bash -lc "export NVM_DIR=\"\$HOME/.nvm\"; [[ -s \"\$NVM_DIR/nvm.sh\" ]] && . \"\$NVM_DIR/nvm.sh\"; command -v nvm >/dev/null 2>&1 && nvm use 20 >/dev/null; cd '$ROOT_DIR/frontend' && exec npm run dev -- --host 0.0.0.0" \
    >"$FRONTEND_LOG_FILE" 2>&1 &
  echo $! >"$FRONTEND_PID_FILE"
  echo "Started frontend (pid $(<"$FRONTEND_PID_FILE")). Log: $FRONTEND_LOG_FILE"
}

stop_process() {
  local name="$1"
  local pid_file="$2"

  if ! [[ -f "$pid_file" ]]; then
    echo "$name is not running."
    return
  fi

  local pid
  pid="$(<"$pid_file")"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    echo "$name PID file was empty and has been removed."
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped $name (pid $pid)."
  else
    echo "$name was not running, cleaning stale PID file."
  fi

  rm -f "$pid_file"
}

show_status() {
  if is_running "$BACKEND_PID_FILE"; then
    echo "Backend: running (pid $(<"$BACKEND_PID_FILE"))"
  else
    echo "Backend: stopped"
  fi

  if is_running "$FRONTEND_PID_FILE"; then
    echo "Frontend: running (pid $(<"$FRONTEND_PID_FILE"))"
  else
    echo "Frontend: stopped"
  fi
}

start_all() {
  start_backend
  start_frontend
}

stop_all() {
  stop_process "Frontend" "$FRONTEND_PID_FILE"
  stop_process "Backend" "$BACKEND_PID_FILE"
}

main() {
  local command="${1:-}"

  case "$command" in
    start)
      start_all
      ;;
    stop)
      stop_all
      ;;
    restart)
      stop_all
      start_all
      ;;
    status)
      show_status
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"