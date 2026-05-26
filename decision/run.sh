#!/bin/bash
# 一鍵執行策略分析（自動使用 .venv）
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "尚未設定環境，先執行: bash setup.sh"
  exit 1
fi

.venv/bin/python strategy.py "$@"
