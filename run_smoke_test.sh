#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "Running FIQS / AllomEq-RAG public smoke test..."
echo

python3 quickstart_smoke.py
