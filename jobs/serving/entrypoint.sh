#!/bin/bash
set -euo pipefail

uvicorn api:app --host 0.0.0.0 --port 8000 &
api_pid=$!

streamlit run dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true &
ui_pid=$!

# bash is PID 1 here, so nothing forwards SIGTERM to the children on `docker stop`:
# without this trap every shutdown ends in a SIGKILL after the grace period.
shutdown() {
    trap - TERM INT
    kill -TERM "$api_pid" "$ui_pid" 2>/dev/null || true
    wait "$api_pid" "$ui_pid" 2>/dev/null || true
    exit 0
}
trap shutdown TERM INT

# Exit as soon as either half dies, so the container's restart policy actually
# applies. Backgrounding both and never waiting would leave a half-dead container
# reporting healthy.
wait -n "$api_pid" "$ui_pid"
status=$?
kill -TERM "$api_pid" "$ui_pid" 2>/dev/null || true
exit "$status"
