#!/usr/bin/env bash
# Load server_env.sh (preferred) or fall back to server_env.example.sh
_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SH_DIR}/server_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_SH_DIR}/server_env.sh"
elif [[ -f "${_SH_DIR}/server_env.example.sh" ]]; then
  echo "WARN: server_env.sh not found; using server_env.example.sh defaults." >&2
  echo "      cp ${_SH_DIR}/server_env.example.sh ${_SH_DIR}/server_env.sh && edit paths" >&2
  # shellcheck source=/dev/null
  source "${_SH_DIR}/server_env.example.sh"
fi
