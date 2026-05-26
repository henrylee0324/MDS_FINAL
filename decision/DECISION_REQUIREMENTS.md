# Hit Song Intelligence — 決策模組需求總結

## 任務定位

在組員完成 **feature selection + song_hotttnesss 預測模型** 後，以 **多限制式 0–1 背包（ILP）** 從「假設尚未發行的新歌」中選出發行組合，並與 **傳統 A&R（各檔期依 ŷ 貪婪）** 對照。

- **候選池**：不用 MSD 百萬曲庫；由 head200 中 **1 首原型** 調參生成 **20 首模擬新歌**
- **目標**：最大化 Σ ŷ（HistGB 預測 `song_hotttnesss`）
- **對照**：傳統各檔 ŷ 貪婪 + FLOP–HOP–TOP 獨立排序（候選池內相對分位）

## PDF §4.2 限制式（全部實作）

| 編號 | 內容 | 實作 |
|------|------|------|
| (1) | max Σ ŷ·x | `knapsack_solver.solve_multi_period` 目標函數 |
| (2) | 各檔 Σ cost·x ≤ C_k | `budget_by_period` |
| (3)(4) | 各檔聲學平均 ≥ T_jk·Σx | `acoustic_min_by_period` + 校準 |
| (5) | L_k ≤ Σx ≤ U_k | `l_min_by_period` / `u_max_by_period` |
| (6) | 每曲最多一檔 | Σ_k x_ik ≤ 1 |
| (7) | x ∈ {0,1} | PuLP Binary |

## 商業參數（本版設定）

| 項目 | 數值 |
|------|------|
| 候選曲數 | 20 |
| 檔期 | Q1 春夏檔 / Q2 秋季檔 / Q3 年末檔 |
| 總預算 | 210（75 + 70 + 65） |
| 各檔 L–U | 2–4 / 2–4 / 1–3 |
| 每首成本 | `COST_BY_PROFILE`（20–60，依情境，與 ŷ 脫鉤） |
| 聲學門檻 | 各檔 tempo / energy / danceability（政策初值 + 候選曲 20% 分位校準） |

## 流程

```
head200 CSV → 資料診斷 → 20 首模擬新歌 → HistGB 預測 ŷ
    → 多檔期 ILP → 傳統貪婪對照 → Excel / CSV / Markdown 報告
```

主程式：`strategy.py`（`bash run.sh`）

## 產出物

| 檔案 | 用途 |
|------|------|
| `outputs/HitSong_策略決策報告.xlsx` | 簡報主檔 |
| `outputs/decision_comparison.csv` | ILP vs 傳統 |
| `outputs/knapsack_multperiod_solution.csv` | ILP 各檔排程 |
| `outputs/決策說明.md` | 中文決策摘要 |
