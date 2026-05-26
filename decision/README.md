# 期末決策：20 首模擬新歌 × 三檔期 × 多限制式背包

對應研討會 PDF §4.2：**預算、檔期發行量、聲學平均門檻、每曲僅排一檔**，並比較 **ILP 最優** vs **傳統 A&R（各檔 ŷ 貪婪）**。

詳細需求見 [DECISION_REQUIREMENTS.md](DECISION_REQUIREMENTS.md)。

## 一鍵執行

```bash
cd decision
bash setup.sh    # 建立 .venv、安裝依賴
bash run.sh      # 執行 strategy.py
```

或：

```bash
.venv/bin/python strategy.py --csv ../data/head200_MSD_with_all_features_categorical_encoded.csv
```

## 主要程式

| 檔案 | 功能 |
|------|------|
| **`strategy.py`** | 主程式：診斷 → 20 首模擬 → 預測 → 多檔期 ILP → 對照 → Excel |
| `knapsack_solver.py` | ILP 求解 + 傳統貪婪 |
| `config_columns.py` | 成本、檔期、聲學門檻、20 組 profile |
| `make_synthetic_new_songs.py` | 1 原型 → 20 首 |
| `score_synthetic_songs.py` | HistGB 預測 ŷ |
| `export_excel.py` | Excel 匯出 |

## 產出（`outputs/`）

| 檔案 | 說明 |
|------|------|
| **`HitSong_策略決策報告.xlsx`** | 簡報主檔（含方法比較、排程、聲學檢核） |
| `decision_comparison.csv` | ILP vs 傳統 |
| `knapsack_multperiod_solution.csv` | ILP 各檔選曲 |
| `決策說明.md` | 中文摘要 |
| `strategy_report.md` | 完整 Markdown 報告 |

## 資料路徑

```
MDS_FINAL/data/head200_MSD_with_all_features_categorical_encoded.csv
```

`paths.py` 也會自動搜尋 repo 上層目錄。
