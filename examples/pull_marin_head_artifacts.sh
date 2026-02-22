#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <submission_id> [cluster] [head_sync_root] [relbench_root] [marin_root]"
  echo "Example: $0 tabpfn-all51-seq-shard0-of-1-20260219-212527-libtpu21-full"
  exit 1
fi

SUBMISSION_ID="$1"
CLUSTER="${2:-us-central1}"
HEAD_SYNC_ROOT="${3:-/tmp/ray/job_artifacts}"
RELBENCH_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MARIN_ROOT="${5:-/home/pc0618/marin}"

CONFIG_FILE="${MARIN_ROOT}/infra/marin-${CLUSTER}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config not found: $CONFIG_FILE"
  exit 1
fi

REMOTE_DIR="${HEAD_SYNC_ROOT%/}/${SUBMISSION_ID}/"
LOCAL_ROOT="${RELBENCH_ROOT}/"

echo "Pulling remote artifacts:"
echo "  cluster=$CLUSTER"
echo "  remote_dir=$REMOTE_DIR"
echo "  local_root=$LOCAL_ROOT"

if command -v rsync >/dev/null 2>&1; then
  (
    cd "$MARIN_ROOT"
    uv run ray rsync-down "$CONFIG_FILE" "$REMOTE_DIR" "$LOCAL_ROOT"
  )
else
  echo "Local rsync not found; using scp fallback."
  HEAD_IP="$(
    cd "$MARIN_ROOT"
    uv run ray get-head-ip "$CONFIG_FILE" | tail -n 1
  )"
  if [[ -z "$HEAD_IP" ]]; then
    echo "Failed to resolve Ray head IP."
    exit 1
  fi

  read -r SSH_USER SSH_KEY < <(
    python3 - "$CONFIG_FILE" <<'PY'
import os
import sys
import yaml

cfg_path = sys.argv[1]
data = yaml.safe_load(open(cfg_path)) or {}
auth = data.get("auth", {}) or {}
user = auth.get("ssh_user", "ray")
key = os.path.expanduser(auth.get("ssh_private_key", "~/.ssh/id_rsa"))
print(user, key)
PY
  )

  scp \
    -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o IdentitiesOnly=yes \
    -o ConnectTimeout=120 \
    -r "${SSH_USER}@${HEAD_IP}:${REMOTE_DIR%/}/." "$LOCAL_ROOT"
fi

echo "Artifact pull complete."
