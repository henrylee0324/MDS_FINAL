# Hit Song Intelligence — 策略分析報告

產生時間: 2026-05-26 11:49
資料來源: `/Users/syuan/Desktop/MDS/head200_MSD_with_all_features_categorical_encoded.csv`

## 1. 資料集診斷

| 項目 | 值 |
|---|---|
| 列數 × 欄數 | 200 × 1754 |
| 目標變數 | `song_hotttnesss` |
| 有效熱度樣本 (>0) | 66 |
| 熱度 mean ± std | 0.4430 ± 0.4006 |
| 曲風 term_* 欄位數 | 1486 |
| 模型特徵數 | 1741 |
| 原型列 index | 0 |
| 原型曲名 / 藝人 | Silent Night / Faster Pussy cat |
| 原型真實熱度 | 0.5218 |

## 2. 模擬新歌（20 首）

由 **1 首原型** 調整 loudness / tempo / duration / year / danceability / energy 與音訊微擾生成，
不作為歷史曲庫直接決策（符合期末「假設新歌」設定）。

## 3. FLOP–HOP–TOP 獨立排序（傳統 A&R）

依預測 ŷ 分級，**不考慮預算**：

| 排名 | ID | 情境 | ŷ | 分級 |
|---:|---|---|---:|---|
| 1 | NEW_SONG_20 | 合成器流行 | 0.6254 | TOP |
| 1 | NEW_SONG_04 | 最新潮流 | 0.6254 | TOP |
| 1 | NEW_SONG_14 | 街頭嘻哈 | 0.6254 | TOP |
| 1 | NEW_SONG_07 | 舞曲場 | 0.6254 | TOP |
| 2 | NEW_SONG_10 | 實驗邊緣 | 0.6254 | TOP |
| 2 | NEW_SONG_19 | 民謠清新 | 0.6254 | TOP |
| 3 | NEW_SONG_16 | 兒童節慶 | 0.6202 | TOP |
| 3 | NEW_SONG_13 | 懷舊金曲 | 0.6202 | TOP |
| 3 | NEW_SONG_01 | 夏日高能量 | 0.6202 | TOP |
| 3 | NEW_SONG_09 | 主流中庸 | 0.6202 | TOP |
| 4 | NEW_SONG_05 | 短歌衝刺 | 0.6202 | HOP |
| 4 | NEW_SONG_17 | 金屬衝擊 | 0.6202 | HOP |
| 4 | NEW_SONG_11 | 電音派對 | 0.6202 | HOP |
| 5 | NEW_SONG_02 | 抒情慢板 | 0.6201 | FLOP |
| 5 | NEW_SONG_12 | 原聲不插電 | 0.6201 | FLOP |
| 5 | NEW_SONG_08 | 民謠感 | 0.6201 | FLOP |
| 5 | NEW_SONG_06 | 長曲敘事 | 0.6201 | FLOP |
| 5 | NEW_SONG_15 | 電影配樂感 | 0.6201 | FLOP |
| 5 | NEW_SONG_18 | 爵士深夜 | 0.6201 | FLOP |
| 5 | NEW_SONG_03 | 復古翻新 | 0.6201 | FLOP |

若只簽 TOP 3（獨立思維）:
NEW_SONG_20, NEW_SONG_04, NEW_SONG_14

## 4. 背包最優決策（全局組合）

| 參數 | 值 |
|---|---|
| 預算 C | 210.0 |
| 最少發行 L | 5 |
| 最多發行 U | 10 |
| loudness 平均下限 | 未啟用 |
| tempo 平均下限 | 未啟用 |

> ⚠️ 聲學門檻已依候選曲 20% 分位數校準（確保 ILP 可行）。 U_MAX 已調整為 10（預算上限）

### 最終選中曲目

| ID | 情境 | ŷ | 成本 | tier |
|---|---|---:|---:|---|
| NEW_SONG_14 | 街頭嘻哈 | 0.6254 | 50 | TOP |
| NEW_SONG_19 | 民謠清新 | 0.6254 | 24 | TOP |
| NEW_SONG_13 | 懷舊金曲 | 0.6202 | 30 | TOP |
| NEW_SONG_02 | 抒情慢板 | 0.6201 | 28 | FLOP |
| NEW_SONG_12 | 原聲不插電 | 0.6201 | 20 | FLOP |
| NEW_SONG_08 | 民謠感 | 0.6201 | 22 | FLOP |
| NEW_SONG_18 | 爵士深夜 | 0.6201 | 26 | FLOP |

**合計** ŷ = 4.3515，成本 = 200，首數 = 7

### 對照：TOP 貪婪（預算內依 ŷ 由高到低）

NEW_SONG_04, NEW_SONG_07, NEW_SONG_14, NEW_SONG_20
 — 合計 ŷ = 2.5017

**Δŷ（背包 − 貪婪）** = 1.8498

## 5. 預算敏感度

| 預算 | 選中首數 | 總 ŷ | 總成本 | 選中 ID |
|---:|---:|---:|---:|---|
| 120 | 5 | 3.1058 | 120 | NEW_SONG_02,NEW_SONG_08,NEW_SONG_12,NEW_SONG_18,NEW_SONG_19 |
| 150 | 6 | 3.7260 | 150 | NEW_SONG_02,NEW_SONG_08,NEW_SONG_12,NEW_SONG_13,NEW_SONG_18,NEW_SONG_19 |
| 180 | 6 | 3.7313 | 177 | NEW_SONG_03,NEW_SONG_08,NEW_SONG_12,NEW_SONG_13,NEW_SONG_19,NEW_SONG_20 |
| 210 | 7 | 4.3515 | 208 | NEW_SONG_08,NEW_SONG_12,NEW_SONG_13,NEW_SONG_16,NEW_SONG_18,NEW_SONG_19,NEW_SONG_20 |
| 240 | 8 | 4.9717 | 240 | NEW_SONG_02,NEW_SONG_08,NEW_SONG_12,NEW_SONG_13,NEW_SONG_14,NEW_SONG_16,NEW_SONG_18,NEW_SONG_19 |

## 6. 多檔期排程（x_ik）

| 檔期 | ID | 情境 | ŷ |
|---|---|---|---:|
| Q3_年末檔 | NEW_SONG_02 | 抒情慢板 | 0.6201 |
| Q1_春夏檔 | NEW_SONG_08 | 民謠感 | 0.6201 |
| Q2_秋季檔 | NEW_SONG_12 | 原聲不插電 | 0.6201 |
| Q3_年末檔 | NEW_SONG_13 | 懷舊金曲 | 0.6202 |
| Q1_春夏檔 | NEW_SONG_14 | 街頭嘻哈 | 0.6254 |
| Q2_秋季檔 | NEW_SONG_18 | 爵士深夜 | 0.6201 |
| Q2_秋季檔 | NEW_SONG_19 | 民謠清新 | 0.6254 |

## 7. 決策建議（簡報用）

1. **科學輸入**：背包使用模型預測 ŷ，非 MSD 歷史曲庫直接排序。
2. **商業輸入**：成本分級 TOP/HOP/FLOP 與預算 C 請與組員對齊後寫入 `config_columns.py` / `StrategyConfig`。
3. **敘事重點**：若 Δŷ > 0，強調「HOP 組合優於全押 TOP」；若為 0，則強調「在約束下已達可行最優」。
4. **限制聲明**：head200 樣本小，CV R² 僅供參考；決策為輔助排序而非保證爆款。

---
產出檔：`outputs/strategy_report.md`、`strategy_summary.csv`、`knapsack_solution.csv`

## 8. 傳統 vs ILP 比較

```
             方法  選中首數     Σŷ   總成本                                                                               選中 ID 聲學門檻全通過 Δŷ vs 傳統
ILP 多限制式背包（本模型）     7 4.3515 200.0 NEW_SONG_02,NEW_SONG_08,NEW_SONG_12,NEW_SONG_13,NEW_SONG_14,NEW_SONG_18,NEW_SONG_19    True   1.8498
傳統 A&R（各檔 ŷ 貪婪）     4 2.5016 172.0                                     NEW_SONG_20,NEW_SONG_19,NEW_SONG_04,NEW_SONG_14    True      0.0
   ILP · Q1_春夏檔     2 1.2455  72.0                                                             NEW_SONG_08,NEW_SONG_14                 
    傳統 · Q1_春夏檔     2 1.2508  70.0                                                             NEW_SONG_20,NEW_SONG_19                 
   ILP · Q2_秋季檔     3 1.8656  70.0                                                 NEW_SONG_12,NEW_SONG_18,NEW_SONG_19                 
    傳統 · Q2_秋季檔     1 0.6254  52.0                                                                         NEW_SONG_04                 
   ILP · Q3_年末檔     2 1.2403  58.0                                                             NEW_SONG_02,NEW_SONG_13                 
    傳統 · Q3_年末檔     1 0.6254  50.0                                                                         NEW_SONG_14                 
```
