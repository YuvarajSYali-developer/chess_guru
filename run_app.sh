#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Starting backend on http://127.0.0.1:8000 ..."
python -m uvicorn backend.app:app --reload --port 8000 --app-dir "$ROOT" &

echo "Starting frontend on http://127.0.0.1:8080 ..."
(cd "$ROOT/frontend" && python -m http.server 8080) &

echo "Waiting for backend to be ready..."
for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

python - <<'PY'
import webbrowser
webbrowser.open("http://127.0.0.1:8080/?v=1")
PY
