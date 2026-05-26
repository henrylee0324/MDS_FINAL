#!/bin/bash
# 一次性環境設定（macOS 無 pip 時用此腳本）
set -e
cd "$(dirname "$0")"

echo ">> 建立虛擬環境 .venv ..."
python3 -m venv .venv

echo ">> 安裝套件 ..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "完成。之後請用："
echo "  source .venv/bin/activate"
echo "  python strategy.py"
echo ""
echo "或不啟動 activate，直接："
echo "  .venv/bin/python strategy.py"
