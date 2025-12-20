#!/bin/bash
# Mouham'IA - Your AI Legal Assistant (محامي + IA)
cd "$(dirname "$0")"
source .venv/bin/activate
PYTHONPATH=src python -m legal_rag.cli "$@"

