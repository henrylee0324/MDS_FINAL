# Million Song Dataset：藝人熱度的線性模型實驗

以 Million Song Dataset (MSD) 預處理版本（`data/MSD_with_all_features.db`，約 12 GB SQLite）為起點，以線性模型為主、非線性模型為對照，預測 `artist_hotttnesss`（artist-level）；並從 MSD 官方 H5 取回 `song_hotttnesss` 補做 song-level 預測。

整條 pipeline 拆成 15 個可重跑的腳本（artist-level: 1–9，song-level 延伸: 10–15），每步的程式碼放在 `analysis/`，數值結果存成 CSV 在 `analysis/results/<step>/`。Song-level 部分原先在 `song-hotness` 分支獨立發展，已合回 `main`。

---

## 執行摘要（給技術讀者的全貌）

### 研究設計

兩條故事線各自得到一條「誠實的天花板」：

| 任務 | 樣本單位 | CV | 對照基準 |
|---|---|---|---|
| **track 1（main）**：預測 `artist_hotttnesss` | 藝人（39,233 列） | 隨機 KFold | step 02 audio-only Ridge R²=0.084 |
| **track 2（song-hotness）**：預測 `song_hotttnesss` | 歌（446,317 列） | **GroupKFold by artist** | step 11 ICC=0.572，43 % within-artist 變異 |

### 全 R² 表（5-fold CV，括號內為 std）

| Track | 配置 | R² |
|---|---|---:|
| 1 | OLS audio only (152) | 0.067 |
| 1 | Ridge α=100 audio only | 0.084 |
| 1 | Ridge strict（185，可解釋線性 baseline） | **0.237** |
| 1 | HistGB strict + top-30 genre | 0.357 |
| 1 | HistGB strict + top-1000 genre | **0.428** |
| 1 | HistGB full + top-1000 genre + tl(12) | 0.464 |
| 1 | CatBoost full + 7,538 tag text features（含 leakage） | **0.470** |
| 2 | HistGB strict_song + top-1000，GroupKFold | **0.290** |
| 2 | HistGB full_song + top-1000 + tl，GroupKFold | 0.317 |
| 2 | CatBoost strict_song + text(7538)，GroupKFold | 0.286 |

### 五個最重要的方法論教訓

1. **音訊特徵是裝飾**——152 個音訊特徵的線性 R²（0.067）只到「5 個 metadata 特徵」（year/duration/n_tracks/n_genres/year_known_ratio）的 R²（0.258）的四分之一。即使換到非線性與 song-level，整個音訊 block 在 GroupKFold 下也只貢獻 ΔR²=0.028。

2. **Vocabulary > Algorithm**——artist-level 從 R²=0.36 推到 0.43 的 +0.07 改進，**95 % 來自 genre vocabulary 從 30 開到 1000+，5 % 來自 HistGB → CatBoost 換模型**。step 09 vocab scaling 直接證實。

3. **Leakage 檢查必須是 cluster-aware**——`n_tracks` 與 `n_genres` 是 popularity 的反向因果代理（藝人紅 → 被收錄多歌 / 被分多 tag），artist-level KFold 看到的 R²=0.42 中有 0.06 是這類 leak；step 14 進一步發現「Echo Nest 給更紅藝人更多 tag」**還隱藏在 vocabulary 中**——一旦改 GroupKFold by artist，CatBoost 全 7,538 tag 的優勢直接消失，與 HistGB top-1000 同等水平。

4. **「Within-artist 變異」是 song-level 預測的真正挑戰**——step 11 量化此變異佔 43 %（ICC=0.572）。step 13 song-level R²=0.29 對照「藝人均值代填」理論上界 0.244，多出來的 0.046 才是 per-track 特徵真正提供的訊號。

5. **單一最強特徵是 `year_per_track`**——step 15 permutation importance 顯示這個欄位 ΔR²=0.112，**比 152 個音訊特徵整塊（0.028）大 4 倍**。Echo Nest 演算法有強烈年代偏差。

### 三個誠實的「天花板」

| 場景 | R² | 用途 |
|---|---:|---|
| **可解釋線性**（Ridge on strict 185） | 0.24 | 寫報告用 |
| **預測藝人 hotness**（HistGB strict + top-1000） | 0.43 | 對「沒見過的藝人」的最佳預測 |
| **預測單首歌 hotness 對未見藝人**（HistGB GroupKFold） | 0.29 | 部署到產品的真天花板 |

### 結論一句話

> **「藝人/歌曲紅不紅」主要由曝光、年代、曲風標籤驅動，音訊本身只是次要訊號**——這個結論被 5 個獨立檢驗（VIF、PCA、Lasso、CatBoost text、permutation importance）相互驗證；不論換目標（artist→song）、換模型（OLS→Ridge→HistGB→CatBoost）、換 vocabulary（30→7,538）、換 CV（KFold→GroupKFold），這個排序都不變。

詳見 [一頁摘要（給非技術讀者）](#一頁摘要給非技術讀者) 與下方各章節。

---

## 一頁摘要（給非技術讀者）

### 我們在問什麼問題？

> **「光聽一首歌的聲音」能不能猜出這位藝人有多紅？**

「紅」這裡用一個 0 到 1 的數字（Echo Nest 公司的 `artist_hotttnesss` 演算法分數）衡量。

### 我們手上的資料

- 100 萬首歌的音訊特徵 → 聚合到 39,292 位藝人
- 每位藝人有 **216 個從聲音計算出來的數值**（音色、頻率組成、和弦結構等）
- 加上 **基本資訊**：年代、歌曲長度、被收錄了幾首歌、曲風標籤

### 四個最重要的發現

| | 發現 | 數字 | 白話 |
|---|---|---|---|
| 1 | 光用聲音預測，效果很差 | R² = 0.07 | 152 個聲音特徵加起來只能解釋「紅不紅」變化的 **7 %** |
| 2 | 加入年代、歌曲數等基本資訊後大幅進步 | R² = 0.42 | 但這 0.42 裡有「作弊成分」（見下） |
| 3 | 拿掉作弊特徵後的誠實天花板 | R² = 0.36（複雜模型）<br/>R² = 0.24（簡單模型） | 對沒被收錄過的新藝人，這是用「簡化版曲風標籤」能達到的上限 |
| 4 | 用 **CatBoost + 全部 7,538 個曲風標籤** | R² = 0.47（含作弊）<br/>R² = 0.43（誠實版） | 之前只用最熱門的 30 個，扔掉了 99.6 % 的標籤資訊 |
| 5 | 進一步驗證：HistGB 配 top-1000 multi-hot 也能達 R²=0.43 | — | **CatBoost 的功勞 95 % 是 vocabulary，5 % 是演算法**——換大字彙比換模型重要 6–10 倍 |
| 6 | （延伸分支）拿到 song_hotttnesss 後做 song-level + GroupKFold | R² = **0.29**（strict）| 對「沒見過的新藝人預測單首歌紅度」這個更嚴格的任務，誠實天花板比 artist-level 的 0.43 低，但這不是退步——是任務變嚴格 |
| 7 | 進一步驗證：在 GroupKFold 下，CatBoost + 7,538 tags 沒比 HistGB + top-1000 好 | R² 仍 0.29 | **artist-level 看到的「CatBoost + 全 vocabulary +0.07」其中大半是 leakage**——一旦公平評估就消失 |

> **R² 是什麼？** 一個 0 到 1 的數字。0 = 跟亂猜一樣，1 = 完美預測，0.5 = 解釋了一半的變化。

### 為什麼會有「作弊成分」

- 「藝人在 MSD 裡有 100 首歌」這件事，**是因為他紅 → 才被收得多**，而不是「歌多 → 讓他紅」
- 用「歌曲數」當特徵看似很有效（R² 從 0.07 衝到 0.42），但對「還沒紅、還沒被收錄」的新藝人根本拿不到這個特徵
- 把這類「事後才有的特徵」拿掉重跑，才是真正能對新資料生效的天花板

### 一句話結論

> **「藝人紅不紅」主要不是被「歌好不好聽」決定的，而是被「曲風標籤、曝光、年代」決定的。** 把 152 個音訊特徵全部加起來，比不上**單一個欄位的 7,538 個曲風標籤文字**。

### 想看技術細節？

- 每個步驟都分成「**白話版**」與「**技術細節**」兩段
- 看到看不懂的術語，跳到最後的「[17. 名詞解釋](#17-名詞解釋)」
- 想複製整個實驗，看「[15. 重現方式](#15-重現方式)」

---

## 1. 任務設定

- **目標變數**：`artist_hotttnesss`（Echo Nest 演算法分數，0–1，少數略大於 1）
- **樣本單位**：藝人（per-`artist_id`）
- **過濾**：`artist_hotttnesss > 0`，排除 `-1`（無資料）與 `0`（缺失）兩種 sentinel
- **聚合**：所有特徵在 SQL 端用 `AVG()` 聚合到藝人層級

> 註：MSD 原版有歌曲層級的 `song_hotttnesss`，但本 DB 在預處理時已被移除；只能以 `artist_hotttnesss` 作為熱度代理。

> 註：8.4 % 的藝人（3,312 / 39,292）在不同歌曲間 `artist_hotttnesss` 不完全一致（median spread 0.028，最大 0.48），原因可能是 Echo Nest 在不同收錄時間取的快照。本實驗以 `AVG` 聚合，已知並接受此瑕疵。

---

## 2. 方法論流程

```
                        ┌─ 01_aggregate_features.py
data/MSD_*.db  ────────►│   SQL GROUP BY artist_id, AVG(features)
                        │   parse `term` → top-30 genre multi-hot
                        └─► cache/artist_audio_agg.pkl  (39,292 × 219)
                            cache/artist_extra.pkl     (39,292 × 35)

                        ┌─ 02_diagnostics_vif_pca.py
audio cache ───────────►│   manual prune duplicates → 152 audio feats
                        │   VIF (diag of inv corr) + PCA scree
                        │   audio-only OLS / Ridge / Lasso / PCA-OLS
                        └─► results/02_diagnostics_vif_pca/vif.csv, pca_explained_variance.csv,
                            results/02_diagnostics_vif_pca/benchmark_audio_only.csv

                        ┌─ 03_extended_baselines.py
both caches ───────────►│   audio + numeric(5) + genre(30) = 187 feats
                        │   B: Ridge across feature subsets
                        │   D: HistGB default/tuned, RandomForest
                        └─► results/03_extended_baselines/benchmark_extended.csv

                        ┌─ 04_feature_importance.py
both caches ───────────►│   Ridge standardized coefficients (full set)
                        │   HistGB tuned + permutation importance
                        └─► results/04_feature_importance/ridge_coefs_full.csv,
                            results/04_feature_importance/perm_importance_full.csv,
                            results/04_feature_importance/importance_by_category.csv

                        ┌─ 05_leakage_check.py
both caches ───────────►│   progressive ablation of n_tracks, n_genres,
                        │     year_known_ratio
                        │   strict-set Ridge coefs + perm importance
                        └─► results/05_leakage_check/benchmark_leakage.csv,
                            results/05_leakage_check/ridge_coefs_strict.csv,
                            results/05_leakage_check/perm_importance_strict.csv
```

---

## 3. 步驟 01：藝人層級聚合

> **白話版**：原始資料是「每首歌一列」（共 100 萬列），但我們要預測的是「藝人有多紅」（藝人層級的數字）。所以把同一位藝人所有歌的音訊特徵取平均，變成「每位藝人一列」。最後得到 **39,292 位藝人** 的資料。同時把曲風標籤（comma-separated 字串）拆解成 30 個最常見的曲風指示變數（有/沒有此標籤 = 1/0）。

`analysis/01_aggregate_features.py`

- 對 `merged_partition1` 按 `artist_id` `GROUP BY`，所有 216 個音訊特徵取 `AVG()`
- 過濾 `CAST(artist_hotttnesss AS REAL) > 0`
- `term` 欄是逗號分隔的 genre 標籤（同藝人內所有歌相同），全集 7,544 個唯一 tag，取 **top 30** multi-hot
- `year_mean` 只對 `year > 0` 的歌計算（47 % 的歌 `year=0` 為缺失）
- 額外加入 `year_known_ratio`（藝人歌曲中 year 已知的比例）、`n_tracks`、`n_genres`

| 輸出 | 形狀 | 內容 |
|---|---|---|
| `cache/artist_audio_agg.pkl` | 39,292 × 219 | artist_id + hotness + n_tracks + 216 audio features |
| `cache/artist_extra.pkl` | 39,292 × 35 | artist_id + 4 numeric + 30 genre indicators |

`hotness` 分佈：mean = 0.366，std = 0.088，min 0.011，max 1.062。
`n_tracks` per artist：median 16，mean 24，max 208（heavy tail）。

---

## 4. 步驟 02：多重共線性診斷 + 純音訊 baseline

> **白話版**：先檢查「152 個音訊特徵」是不是真的 152 個獨立量測。**結論：不是。** 它們之間有大量重複（許多特徵其實在量同一件事的不同形式），本質上只有約 30 個獨立方向。然後用最簡單的線性模型測「光靠音訊」能預測得多準——**答案是 R²=0.08，很差**。
>
> 為什麼會重複？例如「Chroma A、A#、B、C…G#」這 12 個音名特徵都在描述同一段音樂的調性，不管哪個音樂稍微變化，這 12 個會一起變動，本質上是「同一個訊號的 12 個視角」。

`analysis/02_diagnostics_vif_pca.py`

### 4.1 手動修剪重複特徵

| 修剪原因 | 欄位 | 數量 |
|---|---|---:|
| `Method_of_Moments_*` 是 `Area_Method_of_Moments_*` 的子集合 | drop `Method_of_Moments_*` | −10 |
| Marsyas 的 `Mem20_MFCC0..12 × {Mean,Std} × {Mean,Std}` 與 `MFCC_Overall_*` 在量同一件事 | drop marsyas MFCC slice | −52 |
| 全為 NaN 的藝人（`audio.dropna()`） | −59 列 | — |
| 標準化後 zero-variance 欄 | −2 欄 | — |

剩下 **39,233 列 × 152 個音訊特徵** 進入後續分析。被 zero-variance 過濾掉的 2 欄為 `LPC_Overall_Standard_Deviation_10` 與 `LPC_Overall_Average_10`（10 階 LPC 係數聚合到藝人層級後沒變化）。完整過濾紀錄見 `results/06_record_feature_sets/feature_sets_filtering.csv`。

### 4.2 VIF（用 inverse correlation 對角線一次算）

| 指標 | 值 |
|---|---|
| Correlation matrix condition number | 3.52 × 10⁹（病態矩陣）|
| VIF median | 182.47 |
| VIF max | 9.45 × 10⁶ |
| VIF > 10 | 122 / 152（80 %）|
| VIF > 1,000 | 57 / 152 |
| VIF > 10⁶ | 12 / 152 |

最嚴重的是 `Area_Method_of_Moments_Overall_*_{4..9}`——這是相鄰頻段的同類矩，本來就互相能線性表達。完整 VIF 表見 `results/02_diagnostics_vif_pca/vif.csv`。

### 4.3 PCA scree

| 累積 variance | 需 PC 數 |
|---:|---:|
| 50 % | 3 |
| 80 % | 9 |
| 90 % | 18 |
| 95 % | 30 |
| 99 % | 58 |

152 維特徵的本質維度約 30——與 VIF 結果一致。完整數據見 `results/02_diagnostics_vif_pca/pca_explained_variance.csv`。

### 4.4 純音訊 baseline（5-fold CV，shuffle=True，random_state=42）

| 模型 | R² |
|---|---:|
| OLS（全 152 維）| 0.067 |
| Ridge α = 1 | 0.072 |
| Ridge α = 10 | 0.081 |
| **Ridge α = 100** | **0.084** ← 線性最佳 |
| Lasso α = 0.001 | 0.074 |
| Lasso α = 0.01 | 0.019（過度懲罰）|
| OLS top-3 PCs | 0.021 |
| OLS top-9 PCs | 0.051 |
| OLS top-18 PCs | 0.062 |
| OLS top-30 PCs | 0.071 |
| OLS top-58 PCs | 0.079 |
| OLS top-100 PCs | 0.081 |
| OLS top-152 PCs | 0.067（= 全特徵 OLS）|

**結論**：消除共線性（Ridge / PCA / Lasso）只把 R² 從 0.067 提升到 0.084。**音訊資訊量本身就有限**，方法論上的處理只是邊際。

完整表見 `results/02_diagnostics_vif_pca/benchmark_audio_only.csv`。

---

## 5. 步驟 03：擴展特徵集（B + D）

> **白話版**：步驟 02 的結論是「光靠聲音預測不準」。那加上其他資訊（年代、歌長、曲風標籤等）會不會更好？這一步驗證：
>
> - **加入 5 個基本欄位**（年代、時長、`year_known_ratio`、歌曲數、標籤數）：R² 從 0.07 跳到 **0.26**
> - **再加入 30 個熱門曲風標籤**：R² 到 **0.31**（線性模型最佳）
> - **改用更聰明的非線性模型**（HistGradientBoosting）：R² 到 **0.42**（整體最佳）
>
> 也就是說：**5 個基本資訊欄位（R²=0.26）的預測力比 152 個音訊特徵（R²=0.07）高 3.6 倍**。聲音不是預測藝人熱度的主要訊號。

`analysis/03_extended_baselines.py`

特徵組成：**152 audio + 5 numeric + 30 genre = 187**。
Ridge pipeline：audio + numeric 區塊 median-impute → standardize；genre 區塊 0/1 passthrough；用 `RidgeCV(alphas=[0.1, 1, 10, 100, 1000])` 自動選 α。
Tree models：HistGB 對 NaN 原生支援；RandomForest 用 median-impute。

### 5.1 B：Ridge 線性遞進（5-fold CV R²）

| 特徵組合 | n_feat | R² |
|---|---:|---:|
| audio only | 152 | 0.074 |
| audio + numeric | 157 | 0.260 |
| audio + genre | 182 | 0.195 |
| **audio + numeric + genre** | **187** | **0.310** ← 線性最佳 |
| non-audio only | 35 | 0.304 |
| numeric only | 5 | 0.258 |
| genre only | 30 | 0.154 |

**5 個 numeric 特徵**（year, duration, year_known_ratio, n_tracks, n_genres）的 Ridge R²（0.258）已勝過 152 個音訊特徵 R²（0.074）。

### 5.2 D：非線性模型（同 187 維）

| 模型 | R² |
|---|---:|
| HistGB default（300 iter, lr=0.05, depth=6） | 0.420 |
| **HistGB tuned**（600 iter, lr=0.03, depth=8, l2=0.1, min_samples_leaf=20） | **0.422** |
| HistGB tuned（audio only 152） | 0.209 |
| RandomForest（200 trees, depth=20） | 0.395 |

線性 → 非線性差距：0.310 → 0.422，**+11 個百分點**為 nonlinearity / interactions 的貢獻。

完整表見 `results/03_extended_baselines/benchmark_extended.csv`。

---

## 6. 步驟 04：特徵重要度（在 187 維 full set 上）

> **白話版**：拆解上一步的最佳模型，看到底是「哪些特徵」在驅動預測。**結論非常清楚**：
>
> - 排名第 1：`n_tracks`（這位藝人在 MSD 裡有幾首歌）
> - 排名第 2：`n_genres`（這位藝人的曲風標籤有幾個）
> - 排名第 3、4：`year_known_ratio`、`year_mean`（年代相關）
> - 排名第 5 之後：才開始出現曲風指示（pop, alternative）
> - **音訊特徵幾乎不在前 20**
>
> **5 個基本欄位的累計重要度 ≈ 30 個曲風標籤 + 152 個音訊特徵 加起來的總和**。這就是為什麼下一步要做「leakage check」——前兩名其實有「作弊嫌疑」。

`analysis/04_feature_importance.py`

### 6.1 Ridge 標準化係數（α 由 RidgeCV 選為 1000）

| Top 10 正向（推升 hotness） | coef |
|---|---:|
| n_tracks | +0.026 |
| n_genres | +0.017 |
| year_known_ratio | +0.016 |
| genre_pop | +0.014 |
| genre_alternative | +0.012 |
| genre_electronica | +0.008 |
| genre_rock | +0.007 |
| genre_indie | +0.006 |
| Spectral_Flux_Std | +0.006 |
| year_mean | +0.005 |

| Top 10 負向（拉低 hotness） | coef |
|---|---:|
| genre_world | −0.013 |
| genre_alternative_rock | −0.010 |
| genre_house | −0.009 |
| genre_synthpop | −0.008 |
| genre_united_states | −0.007 |
| Root_Mean_Square_Std | −0.006 |
| marsyas ZeroCrossings_Std | −0.005 |
| genre_trance | −0.004 |
| Spectral_Flux_Avg | −0.004 |
| genre_pop_rock | −0.004 |

完整係數見 `results/04_feature_importance/ridge_coefs_full.csv`。

### 6.2 HistGB Permutation Importance（top 15）

| ΔR² 下降 | feature |
|---:|---|
| 0.186 | **n_tracks** |
| 0.133 | **n_genres** |
| 0.056 | year_known_ratio |
| 0.037 | year_mean |
| 0.014 | genre_pop |
| 0.014 | genre_alternative |
| 0.008 | genre_alternative_rock |
| 0.006 | genre_house |
| 0.006 | Area_MoM_Std_2 |
| 0.005 | genre_american |
| 0.005 | genre_world |
| 0.004 | marsyas Flux_Std |
| 0.003 | genre_synthpop |
| 0.003 | genre_united_states |
| 0.003 | genre_electronica |

完整數值見 `results/04_feature_importance/perm_importance_full.csv`。

### 6.3 依特徵類別累計 importance

| Category | sum | count | per-feat |
|---|---:|---:|---:|
| **numeric** | **0.414** | 5 | 0.0827 |
| genre | 0.072 | 30 | 0.0024 |
| audio:marsyas | 0.015 | 72 | 0.00021 |
| audio:Area_MoM | 0.011 | 20 | 0.00055 |
| audio:MFCC_simple | 0.004 | 26 | 0.00015 |
| audio:LPC | 0.003 | 18 | 0.00019 |
| audio:lowlevel | 0.003 | 16 | 0.00018 |

5 個 numeric 特徵的累計重要度 ≈ 30 個 genre + 152 個 audio 的總和。

完整表見 `results/04_feature_importance/importance_by_category.csv`。

---

## 7. 步驟 05：Leakage check（誠實的預測天花板）

> **白話版**：步驟 04 顯示「歌曲數」是排名第 1 的特徵。但仔細想：**為什麼一位藝人在 MSD 裡會有 100 首歌？因為他紅到值得被收錄這麼多。** 這是「果」不是「因」——對一位「還沒紅、還沒被收錄」的新藝人，我們根本沒有「歌曲數」這個資訊。
>
> 這種「**事後才能取得，無法用於真實預測**」的特徵叫做 **資料洩漏（data leakage）**。包含洩漏特徵的模型在訓練資料上看似很準，但對真實新案例毫無用處。
>
> 把這類特徵（`n_tracks`、`n_genres`）拿掉重新訓練，得到 **誠實的天花板**：
>
> - 線性模型：R² 從 0.32 → **0.24**
> - 非線性模型：R² 從 0.42 → **0.36**
>
> 也就是 0.42 中有 **0.06（14 %）是「假的」預測力**。
>
> 還有一個小驚喜：拿掉這些主導特徵後，**音訊特徵終於浮上檯面**——`Spectral Variability`、`Area Method of Moments`、`Marsyas Chroma PeakRatio` 變成模型最看重的音訊類別。

`analysis/05_leakage_check.py`

### 7.1 為什麼要做 leakage check

`n_tracks` 和 `n_genres` 是「**藝人之所以紅 → MSD 收錄了他更多歌、Echo Nest 給他更多 tag**」的反向因果——對未被收錄的新藝人**根本拿不到這兩個特徵**。把它們留在模型裡會虛報 R²。

### 7.2 漸進式移除（5-fold CV R²）

| Config | n_feat | Ridge | HistGB |
|---|---:|---:|---:|
| full | 187 | 0.324 | **0.422** |
| − n_tracks | 186 | 0.257 | 0.390 |
| − n_genres | 186 | 0.312 | 0.390 |
| **− n_tracks − n_genres**（strict） | **185** | **0.237** | **0.357** |
| − strict + year_known_ratio | 184 | 0.196 | 0.344 |
| audio + genre only | 182 | 0.196 | 0.311 |
| audio only | 152 | 0.081 | 0.209 |

- 拿掉 `n_tracks` 對 Ridge 衝擊最大（−0.067）
- `n_genres` 對 Ridge 影響輕微（−0.012），但對 HistGB 也是 −0.032
- 兩個都拿掉：**Ridge 從 0.32 → 0.24，HistGB 從 0.42 → 0.36**
- 0.42 中約 0.06（14 %）來自 leakage，不是真實預測力

完整表見 `results/05_leakage_check/benchmark_leakage.csv`。

### 7.3 Strict set 上 Ridge 係數重新洗牌

由於 `n_tracks` / `n_genres` 不再 dominate，RidgeCV 選了 α=1（而非 full set 的 α=1000），音訊特徵終於浮上檯面：

| Strict Ridge top 10 正向 | coef |
|---|---:|
| Spectral_Variability_Avg | +0.070 |
| Area_MoM_Std_5 | +0.048 |
| marsyas PeakRatio_Chroma_F | +0.046 |
| marsyas PeakRatio_Chroma_F# | +0.046 |
| marsyas PeakRatio_Chroma_G# | +0.042 |
| marsyas PeakRatio_Chroma_A# | +0.039 |
| marsyas PeakRatio_Chroma_G# | +0.035 |
| marsyas PeakRatio_Chroma_D | +0.032 |
| marsyas PeakRatio_Min_Chroma_A | +0.032 |
| Area_MoM_Avg_1 | +0.028 |

完整係數見 `results/05_leakage_check/ridge_coefs_strict.csv`。

### 7.4 Strict set permutation importance（top 15）

| ΔR² | feature |
|---:|---|
| 0.094 | year_known_ratio |
| 0.034 | Area_MoM_Std_2 |
| 0.031 | genre_pop |
| 0.026 | genre_alternative |
| 0.019 | year_mean |
| 0.013 | genre_alternative_rock |
| 0.012 | genre_electronic |
| 0.012 | Area_MoM_Std_7 |
| 0.010 | genre_american |
| 0.010 | marsyas Flux_Std |
| 0.009 | genre_rock |
| 0.009 | genre_electronica |
| 0.007 | genre_house |
| 0.006 | marsyas Flux_Std (Mean_Acc5_Std_Mem20) |
| 0.005 | genre_folk |

### 7.5 Strict set 類別重要度（與 full 對比）

| Category | strict sum | full sum | strict / full |
|---|---:|---:|---:|
| genre | 0.152 | 0.072 | 2.1× |
| numeric (3) | 0.118 | 0.414 | 0.29× |
| audio:Area_MoM | 0.049 | 0.011 | 4.5× |
| audio:marsyas | 0.034 | 0.015 | 2.3× |
| audio:MFCC_simple | 0.011 | 0.004 | 2.8× |
| audio:lowlevel | 0.006 | 0.003 | 2.0× |
| audio:LPC | 0.006 | 0.003 | 2.0× |

移除 leak features 後，**音訊類別的相對重要度全部變成原本的 2–4 倍**。

完整數值見 `results/05_leakage_check/perm_importance_strict.csv`。

---

## 8. 步驟 06：每個跑法用到的特徵集記錄

> **白話版**：上面跑了 22 種「特徵組合 + 模型」配置（例如「audio only」、「strict_drop_both」等）。為了日後能回溯「**這個 R² 數字是用哪些特徵跑出來的**」，把每個 configuration 用到的特徵列表都顯式存成 CSV。日後若想驗證或重做某個結果，可以直接從這些檔案查到對應的特徵清單。

`analysis/06_record_feature_sets.py`

把以上 5 步驟裡每一個 modeling configuration 用到的特徵列表都顯式輸出，以便回溯「**這次的 R² 是哪些特徵跑出來的**」。

| 輸出檔案 | 內容 |
|---|---|
| `06_record_feature_sets/feature_sets_filtering.csv` | 216 個原始音訊特徵的過濾履歷：`in_raw_216` → `in_after_manual` → `in_after_zerovar`，每個特徵都標出在哪一階段被砍掉 |
| `06_record_feature_sets/feature_sets_summary.csv` | 22 個 configuration 的摘要：`script, config, n_features, note`（note 列出各類別特徵數）|
| `06_record_feature_sets/feature_sets_long.csv` | 長表：每個 (config, feature) 配對一列；可篩選 `df[df.config=='strict_drop_both']` 取出該跑法用的全部特徵 |
| `06_record_feature_sets/feature_sets_wide.csv` | 寬表：每個 config 一欄、每個 feature 一列、cell = 0/1 表示是否使用，方便試算表觀察跨 config 的差異 |

涵蓋的 22 個 configuration：

| Script | 設定 | n_features |
|---|---|---:|
| 02 | `audio_only_152` | 152 |
| 02 | `OLS_top{3,9,18,30,58,100}_PCs` | （PC 為投影，非原始特徵）|
| 03 | `audio_only`, `audio_plus_numeric`, `audio_plus_genre`, `full_187`, `non_audio_only`, `numeric_only`, `genre_only` | 5 ~ 187 |
| 04 | `full_187` | 187 |
| 05 | `full`, `drop_n_tracks`, `drop_n_genres`, `strict_drop_both`, `drop_year_known_ratio`, `audio_plus_genre`, `audio_only` | 152 ~ 187 |

---

## 9. 特徵選擇方法論說明

> **白話版**：本實驗的特徵選擇 **沒有使用任何 data-driven 的演算法**（例如 RFE、Lasso active set、互資訊過濾）。所有的篩選都是基於「特徵名稱看得出重複」、「物理上零變異」、「人為決定 cut-off」、「因果反推 leakage」這四種判斷。這是刻意的設計選擇——優先保留可解釋性與計算速度，但代價是「我們無法保證留下的就是最佳子集」。

### 9.1 已使用的篩選方法（依執行順序）

| # | 方法 | 性質 | 篩了什麼 | 數量變化 | 出處 |
|---:|---|---|---|---:|---|
| 1 | **領域知識：刪重複特徵家族** | 判斷性 | `Method_of_Moments_*`（10 欄）⊂ `Area_Method_of_Moments_*`（20 欄） | 216 → 206 | 步驟 02 |
| 2 | **領域知識：刪重複統計變換** | 判斷性 | Marsyas `Mem20_MFCC0..12 × {Mean,Std} × {Mean,Std}` = 52 欄 ≈ `mfcc_features` 的 26 欄 | 206 → 154 | 步驟 02 |
| 3 | **機械過濾：零變異** | 客觀 | `LPC_Overall_Standard_Deviation_10`、`LPC_Overall_Average_10`（藝人聚合後常數） | 154 → 152 | 步驟 02 |
| 4 | **頻率截斷：top-N** | 啟發式 | 7,544 個 genre tag 中保留出現頻率前 30 名做 multi-hot；其餘 7,514 個丟掉 | +30 | 步驟 01 |
| 5 | **手工選擇非音訊欄** | 判斷性 | 5 個 numeric：`year_mean, duration_mean, year_known_ratio, n_tracks, n_genres` | +5 | 步驟 01 |
| 6 | **因果推理：刪 leakage** | 判斷性 | `n_tracks`、`n_genres`（popularity proxy） | 187 → 185 | 步驟 05 |

最終特徵集合：

- **方法論天花板版本**（含 leakage 特徵）：152 audio + 5 numeric + 30 genre = **187**
- **誠實版（部署用）**：移除 `n_tracks`、`n_genres` → **185**

### 9.2 各篩選方法的判斷依據

#### 9.2.1 為什麼判斷「家族重複」？

兩個依據，都是「看名字 + 看維度」推斷的：

- `Method_of_Moments_Overall_*_{1..5}`（10 欄）與 `Area_Method_of_Moments_Overall_*_{1..10}`（20 欄）共用「Method_of_Moments」與「Overall」前綴，僅維度差異——典型「子集擴充」命名模式。
- 「MFCC」一詞在 marsyas 表（52 欄，巢狀聚合）與 mfcc_features 表（26 欄，整段平均）同時出現，**兩處都在量同一概念，只是聚合方式不同**。實證：步驟 02 跑出的 PC1 解釋 35 % 變異，loading 集中在 chroma 系列——驗證了「同一訊號被切成多視角」的假設。

> **沒有跑統計檢定。** 沒有計算這些欄位之間的實際相關係數來驗證「重複」的程度。

#### 9.2.2 為什麼 top-N 取 N=30？

純粹是 ad-hoc 選擇，理由：

- 30 ≈ 與其他 block 量級相當（5 numeric、152 audio 在同一個量級）
- 能涵蓋主流曲風（pop, rock, electronic, hip hop, jazz...）
- 不會讓 multi-hot 矩陣過度稀疏

> **沒做過 elbow analysis 或 cross-validation 驗證。** N=20 / 50 / 100 沒被嘗試比較。

#### 9.2.3 為什麼「因果推論」就足以判斷 leakage？

`n_tracks`（藝人在 MSD 收錄的歌數）這個特徵的形成過程是：

```
artist 紅 → 商業價值高 → Echo Nest 收錄更多歌 → n_tracks 高
```

也就是 `n_tracks` 是 hotness 的 **下游結果**，不是上游原因。對「未被收錄／未被標籤化的新藝人」，這個特徵根本不存在。**這個推論不需要統計檢定，是定義上的問題**。`n_genres` 同理（標籤是 Echo Nest 演算法事後產生的）。

### 9.3 刻意未使用的篩選方法

| 方法 | 為什麼沒用 | 加上會帶來什麼 |
|---|---|---|
| **VIF 迭代修剪** | Ridge 自然處理共線性，跑迭代 VIF 對結果影響小 | 「最互不冗餘的 30 個音訊代表」清單，可作報告附錄 |
| **Lasso active set** | Lasso 在 step 02 只當預測模型，未記錄哪些係數歸零 | 「Lasso 認為足夠的特徵子集」，數量取決於 α |
| **Mutual Information / target correlation 過濾** | 沒做 univariate filter | 快速砍「與目標完全無關」的特徵；缺點是會誤砍「單獨弱、組合強」的 |
| **Forward / Backward selection** | O(p²) 計算成本太高 | 黃金標準但耗時 |
| **Recursive Feature Elimination (RFE)** | 同上 | 比 forward 快，仍偏貴 |
| **以 permutation importance 篩 top-k 重訓** | 已用 permutation importance 解釋，未拿來迭代 | 留下「真正重要」的子集，可能改善 OOD 表現 |
| **target-stratified CV / time-aware split** | 用 random K-fold | 更嚴格的 OOD 評估，避免 hotness 高/低的藝人在訓練/測試集間洩漏 |

### 9.4 自評與建議延伸

**目前篩選方法的優點**：

- **可解釋**：每個被砍掉的特徵都能講清楚原因
- **快**：不需要訓練多輪模型來決定要保留哪些
- **不過度擬合驗證集**：因為沒用 target 來篩特徵

**目前的缺點**：

- 「家族重複」的判斷僅止於名字推測，**沒有跑相關性矩陣驗證**（雖然事後 PC1 loadings 證實了我們的猜測）
- top-30 genre 的 cut-off 是任意的
- **沒有提供「演算法選出的最佳子集」作為對照**，無法回答「我們選的真的是最好的嗎？」

**若要強化嚴謹度的優先順序建議**：

1. **加跑迭代 VIF**——把 152 個音訊跑 VIF > 10 迭代修剪，看最後留下幾欄、是哪些。可作為步驟 02 的補充表。
2. **記錄 Lasso α=0.001 的 active set**——已知此設定下 R²=0.074，但沒記錄哪些 coef 非零。一個 SQL 級的單行修改即可補。
3. **跑一次 RFECV on Ridge**——以 R² 為目標，自動選 k。耗時較久（~小時級），但給出嚴謹基準。
4. **改用 stratified CV**：依 hotness 分位數分層切 fold，估計更穩健。

---

## 10. 強化嚴謹度（VIF / Lasso / Stratified CV）

> **白話版**：第 9.4 節列了 4 個「該做但沒做」的補強項目。本節跑前 3 項，**結論：原本的手工修剪結果穩健，沒有重大遺漏**。

`analysis/07_rigor_strengthening.py`

### 10.1 Iterative VIF pruning

從 152 個音訊特徵開始，每次砍掉 VIF 最大那一欄，重新計算，直到所有 VIF < 10。

| 指標 | 值 |
|---|---:|
| 起始特徵數 | 152 |
| 迭代次數 | 92 |
| **存活特徵數** | **61** |
| 最終 max VIF | 9.66 |
| 5-fold CV OLS R² | 0.077 |
| 5-fold CV Ridge R² | 0.078 |

對照：原本「全 152 + Ridge α=100」R² = 0.084。

> **結論**：把 152 砍到 61（少 60 %），R² 只掉 0.006。表示**被砍掉的 91 個欄位確實是冗餘的**——既支持「手工修剪不夠激進」，也說明 Ridge 已經有效處理了這些冗餘。**整體預測力幾乎不受影響**。

完整迭代軌跡見 `results/07_rigor_strengthening/vif_iterative_trace.csv`，存活特徵列表見 `results/07_rigor_strengthening/vif_iterative_survivors.csv`。

### 10.2 Lasso active set sweep

對 152 個標準化音訊特徵跑 Lasso，在多個 α 下記錄非零係數：

| α | 非零特徵數 | 5-fold CV R² |
|---:|---:|---:|
| 0.0001 | 82 | 0.083 |
| 0.0005 | 46 | 0.079 |
| **0.001** | **27** | **0.074** |
| 0.005 | 10 | 0.047 |
| 0.01  | 4 | 0.019 |

> **結論**：Lasso 認為「足夠用」的特徵約 27 個（α=0.001），R²=0.074；與「全 152 Ridge」R²=0.084 只差 0.010。**確認音訊特徵的有效自由度約在 30–80 之間**——與 PCA 的「95 % variance 需 30 個 PC」、VIF 砍剩 61 個的數字相互印證。

完整 active sets 見 `results/07_rigor_strengthening/lasso_active_sets.csv`（每個 α 都列出對應的特徵清單）。

### 10.3 Stratified-by-hotness 5-fold CV

把藝人按 hotness 五等分位分層，重新評估 4 個關鍵 configuration：

| Config | Model | Random KFold | Stratified CV | Δ |
|---|---|---:|---:|---:|
| full_187 | Ridge | 0.3095 ± 0.0446 | 0.3094 ± 0.0442 | −0.0002 |
| full_187 | HistGB | 0.4217 ± 0.0018 | 0.4218 ± 0.0069 | +0.0002 |
| strict_185 | Ridge | 0.2374 ± 0.0033 | 0.2351 ± 0.0039 | −0.0022 |
| strict_185 | HistGB | 0.3569 ± 0.0039 | 0.3584 ± 0.0059 | +0.0015 |

> **結論**：所有 |Δ| < 0.005，**stratification 沒有改變任何結論**。意思是 random KFold 在本資料上已經是穩健的估計工具——hotness 在 39,233 位藝人中分佈夠均勻，random split 不會偏向特定區間。原本所有 R² 數字都不需要重跑。

完整數值見 `results/07_rigor_strengthening/stratified_cv_benchmark.csv`。

### 10.4 三個補強方法的綜合評估

| 補強方法 | 改變 R² | 改變結論 | 增加什麼新資訊 |
|---|---|---|---|
| Iterative VIF pruning | −0.006 | 否 | 「~60 個 VIF<10 的核心音訊特徵」清單 |
| Lasso α 掃描 | 隨 α 而異 | 否 | 「~27 個 Lasso 認為夠用的特徵」清單 |
| Stratified-by-hotness CV | |Δ| < 0.005 | 否 | 確認原 R² 估計的穩健性 |

> **跨方法交叉印證**：PCA 95 % variance → 30 個 PC；iterative VIF → 61 個存活；Lasso α=0.001 → 27 個 active。三種彼此獨立的方法**都指向「音訊本質自由度在 30–80 之間」**。原本基於命名推斷的「152 個音訊大量冗餘」假設，到此被三個演算法獨立驗證。

---

## 11. CatBoost 與全 vocabulary genre tags

> **白話版**：之前所有實驗都只用「最熱門的 30 個 genre 標籤」，因為 HistGB / Ridge 沒辦法直接吃 7,544 個原始 tag（會 OOM）。CatBoost 的 `text_features` 機制可以原生處理，把全部 tag 攤開來用一次。結果：**HistGB strict (top-30) 0.357 → CatBoost strict + 全部 tags 0.430**（+0.073），**HistGB full 0.422 → CatBoost full + 全部 tags 0.470**（+0.048）。
>
> 這個改善的真正來源不是 CatBoost 演算法比較強，而是**之前丟掉了 99.6 % 的 tag 資訊**（30 / 7,538 = 0.4 %）。

`analysis/08_catboost.py`

### 11.1 4 個配置的 5-fold CV R²

| Config | n cols | tag 範圍 | CV R² | vs HistGB |
|---|---:|---:|---:|---:|
| HistGB strict (top-30，參考) | 185 | 30 | 0.357 | — |
| CatBoost strict (top-30) | 185 | 30 | 0.367 | +0.010 |
| **CatBoost strict + 全 tags** | **156** | **7,538** | **0.430** | **+0.073** ⭐ |
| HistGB full (top-30，參考) | 187 | 30 | 0.422 | — |
| CatBoost full (top-30) | 187 | 30 | 0.428 | +0.006 |
| **CatBoost full + 全 tags** | **158** | **7,538** | **0.470** | **+0.048** ⭐ |

完整數值見 `results/08_catboost/benchmark_catboost.csv`。

### 11.2 CatBoost 本身 vs vocabulary 大小：拆解貢獻

兩個維度交叉拆解（同一行同一個 vocabulary，同一列同一個模型）：

| | top-30 multi-hot | 全部 7,538 tags | Δ from vocab |
|---|---:|---:|---:|
| HistGB strict / full | 0.357 / 0.422 | （HistGB 無法吃文字）| — |
| CatBoost strict / full | 0.367 / 0.428 | 0.430 / 0.470 | **+0.063 / +0.042** |
| Δ from model | +0.010 / +0.006 | — | |

- **換模型（HistGB → CatBoost）**：+0.006 ~ +0.010
- **換 vocabulary（top-30 → 全部 7,538）**：+0.042 ~ +0.073

vocabulary 帶來的改進是換模型的 **6–10 倍**——驗證之前的判斷「真正瓶頸是 feature engineering，不是 algorithm」。

### 11.3 CatBoost 特徵重要度（top 15）

訓練在最佳配置（full + text）上：

| Importance | Category | Feature |
|---:|---|---|
| 32.68 | **genre_text** | **`term_norm`（單一欄包 7,538 個 tags）** |
| 14.73 | numeric | n_tracks |
| 13.15 | numeric | n_genres |
| 7.55 | numeric | year_known_ratio |
| 4.54 | numeric | year_mean |
| 0.76 | numeric | duration_mean |
| 0.71 | audio:marsyas | Chroma_G# (Std × Std) |
| 0.68 | audio:marsyas | Chroma_A# (Std × Std) |
| 0.66 | audio:Area_MoM | Std_2 |
| 0.57 | audio:LPC | Avg_4 |
| 0.56 | audio:marsyas | Chroma_F# (Std × Mean) |
| 0.46 | audio:MFCC_simple | Std_1 |
| 0.42 | audio:LPC | Avg_2 |
| 0.42 | audio:lowlevel | Fraction_Of_Low_Energy_Windows |
| 0.41 | audio:marsyas | Chroma_G# (Std × Mean) |

完整列表見 `results/08_catboost/feature_importance_catboost.csv`。

### 11.4 類別累計重要度

| Category | sum | n columns | per-column |
|---|---:|---:|---:|
| numeric | 40.73 | 5 | 8.15 |
| **genre_text** | **32.68** | **1** | **32.68** |
| audio:marsyas | 11.57 | 72 | 0.16 |
| audio:MFCC_simple | 4.80 | 26 | 0.18 |
| audio:LPC | 4.34 | 18 | 0.24 |
| audio:Area_MoM | 3.04 | 20 | 0.15 |
| audio:lowlevel | 2.85 | 16 | 0.18 |

> **`term_norm` 一個欄位的重要度（32.68）超過全部 152 個音訊特徵的總和（26.59）**。這就是為什麼把 vocabulary 從 30 攤到 7,538 帶來這麼大的提升。

完整類別表見 `results/08_catboost/feature_importance_catboost_by_category.csv`。

### 11.5 ⚠️ 「全部 tags」算 leakage 嗎？

部分算。Echo Nest 演算法給「越紅的藝人」越多細節 tag——所以 tag 數量本身就含有 popularity 訊號。但相較於 `n_tracks` / `n_genres`，**tag 內容（hip hop / jazz / hardcore）比 tag 計數更接近「歌的曲風本質」**。

- 對「Echo Nest 已分析過 tag 的藝人」：CatBoost text **0.470** 是合理的天花板
- 對「完全冷啟動的新藝人」（沒任何 tag）：tag 不存在，**0.470 高估**
- 對「Spotify / Last.fm 等其他平台仍有 tag」的藝人：介於兩者之間（半合法）

`term_norm` 在 strict_185（已移除 n_tracks / n_genres）下仍能達到 **0.430**，比 strict 乾淨的 HistGB 0.357 好了 0.073——這個改進的大部分來自「tag 內容的曲風訊號」，而不只是 tag 數量這個 popularity proxy。

---

## 12. Vocabulary scaling：拆解 CatBoost 的「魔法」

> **白話版**：第 11 節得到「CatBoost + 全部 7,538 個 tags」R²=0.43（strict），比 HistGB top-30 的 0.36 高 0.07。但這 +0.07 是 CatBoost 演算法的功勞，還是「把 vocabulary 從 30 開到 7,538」的功勞？這一節做拆解：在 HistGB（不換模型）上，把 vocabulary 從 30 一路開到 1000 看 R² 怎麼變。

`analysis/09_vocab_scaling.py`

### 12.1 HistGB 在不同 vocabulary 大小下的 R²

| top_N | strict | full | strict Δ vs N=30 |
|---:|---:|---:|---:|
| 30 | 0.358 | 0.422 | — |
| 100 | 0.410 | 0.449 | +0.052 |
| 300 | 0.421 | 0.457 | +0.063 |
| **1000** | **0.428** | **0.462** | **+0.070** |
| **7538（CatBoost 對照）** | **0.430** | **0.470** | **+0.072** |

完整數據見 `results/09_vocab_scaling/benchmark_vocab_scaling.csv`。

### 12.2 解讀

#### **(1) CatBoost 的「魔法」95 % 是 vocabulary，5 % 是內部表徵**

| 比較對象 | strict | full |
|---|---:|---:|
| HistGB top-1000 multi-hot | 0.428 | 0.462 |
| CatBoost text 7,538 tags | 0.430 | 0.470 |
| **Δ（純算法差異）** | **+0.002** | **+0.008** |

換言之：`CatBoost 演算法本身只貢獻 0.002–0.008 R²`，剩下的 0.06–0.07 全是「之前用 top-30 把 99.6 % vocabulary 丟掉」造成的。

#### **(2) Diminishing returns curve**

把 strict 的增量逐段看：

| Step | strict R² 增益 |
|---|---:|
| 30 → 100 | **+0.052** ← 最大跳躍 |
| 100 → 300 | +0.011 |
| 300 → 1000 | +0.007 |
| 1000 → 7538 | +0.002 |

對 strict 而言，**top-300 multi-hot 就能拿到 vocabulary 紅利的 87 %**（從 +0.07 拿到 +0.063），且不需要 CatBoost。

#### **(3) 重大的方法論修正**

- 原本第 7 節說「strict R²=0.36 是誠實天花板」——**這只是「top-30 限制下」的天花板**，不是真正的天花板
- 真正的誠實天花板應該是 **R² ≈ 0.43**（無論用 HistGB top-1000 還是 CatBoost text 都收斂到此）
- 「audio 弱、metadata 強」的結論**仍然成立**，而且**音訊的相對地位更弱**：原本以為 metadata 解釋了 strict 下大部分變異（0.36 中約 0.21 來自 metadata），現在我們知道 metadata 其實能解釋更多（0.43 中約 0.28 來自 metadata）

### 12.3 給之前報告的「打補釘」

| 之前報告寫 | 實際情況 |
|---|---|
| strict 誠實天花板 R²=0.36 | top-30 限制下的天花板；vocabulary 放開後 R²≈0.43 |
| HistGB 是非線性上界 | 在「vocabulary 充足」時，HistGB 與 CatBoost 收斂到相近值 |
| CatBoost 顯著優於 HistGB | 在同樣 vocabulary 下，CatBoost 只比 HistGB 高 0.005–0.010；7,538 tag 的差距並非演算法功勞 |

這是「**先誠實揭露 leakage，再誠實揭露 feature engineering 設限**」的兩段式自我修正。

---

## 13. song_hotttnesss 延伸實驗（branch: `song-hotness`）

> **白話版**：到目前為止所有實驗都用 `artist_hotttnesss` 當目標，因為本 DB 的 `song_hotttnesss` 已被預處理移除。這一節記錄一段在 `song-hotness` 分支上的探索：到 MSD 官網下載 `msd_summary_file.h5`（約 316 MB）把 `song_hotttnesss` 拿回來，重做整條 pipeline。
>
> **核心發現**：song-level 的「對新藝人預測單首歌紅度」誠實天花板 R² ≈ 0.30，比 artist-level 的 0.43 低，但這不是退步——是任務變嚴格的結果。

### 13.1 步驟 10：到網路上抓 song_hotttnesss

`analysis/10_song_hotness_extract.py`

- 從 http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/msd_summary_file.h5 下載（316 MB，重定向到 millionsongdataset.com）
- 用 PyTables 讀 H5 中的 `/metadata/songs`、`/analysis/songs`、`/musicbrainz/songs` 三表，按 row 對齊抽出每首歌的 `song_hotttnesss` 加上 11 個 track-level Echo Nest 分析欄（loudness, tempo, key, mode, time_signature, durations, fade times）

| 指標 | 值 |
|---|---:|
| 1M 首歌中 `song_hotttnesss > 0` | **457,205**（45.7 %） |
| 與我們現有 pipeline tracks 的交集 | 448,552 |
| 至少有 1 首有效歌的 pipeline 藝人 | 30,401 / 39,292 |
| 至少 5 首 | 20,369 / 39,292 |

⚠️ `danceability` 與 `energy` 全部為 0（Echo Nest 從未填寫，已知問題）。

完整 coverage 與 join 報告見 `results/10_song_hotness_extract/coverage_report.csv` 與 `join_with_artist_pipeline.csv`。

### 13.2 步驟 11：診斷——`song_hotttnesss` 多帶多少訊息？

`analysis/11_song_vs_artist_hotness.py`

如果同一藝人各歌的 hotness 幾乎相同，那 `song_hotttnesss` ≈ `artist_hotttnesss + 雜訊`，換目標毫無收穫。實測：

| 統計量 | 值 | 意義 |
|---|---:|---|
| 每藝人歌之間 std 中位數 | 0.087 | 中等差異 |
| Within-artist 變異佔總變異 | **42.8 %** | 不可忽略 |
| Between-artist 變異佔總變異 | 57.2 % | 仍然主導 |
| **ICC**（intraclass correlation） | **0.572** | 「值得做 song-level」 |
| Pearson r(per-artist mean, artist_hot) | 0.62 | 不是同一個東西 |
| 60 % 藝人 \|song_mean − artist_hot\| > 0.05 | — | 兩者普遍有差距 |

**極端案例**：
- ARRH63Y artist_hot=1.08 但有歌只 0.30 → 巨星的冷門曲（dud track）
- ARVGBDX artist_hot=0.38 但有歌達 1.00 → 普通藝人的代表作（hidden hit）

### 13.3 步驟 12：把 track-level Echo Nest 特徵加進 artist-level pipeline

`analysis/12_track_level_features.py`

把 step 10 拿到的 11 個 track-level 欄做 per-artist 聚合（mean 為主，加 mode_fraction 等），加進現有 HistGB top-1000 baseline：

| Config | top-1000 only | + 12 個 track-level | Δ |
|---|---:|---:|---:|
| strict | 0.428 | **0.433** | +0.005 |
| full | 0.462 | **0.463** | +0.002 |

> **結論**：聚合到藝人層級後，per-track 訊號被平均掉，貢獻幾乎為零。提示：這些特徵在 song-level（不被聚合）時應該才會起作用。

### 13.4 步驟 13：song-level pipeline（換目標 + GroupKFold）

`analysis/13_song_level_pipeline.py`

**樣本單位變成「每首歌一列」**：446,317 列 / 30,332 個藝人。
**5-fold 改用 `GroupKFold(artist_id)`**——同藝人的所有歌都進同一 fold，模型測試的是「**完全沒見過的新藝人的歌**」。
特徵：
- 152 audio per-track（**不再聚合**）
- 10 track-level Echo Nest（per-track）
- year, duration（per-track）
- top-1000 genre multi-hot（per-artist 常數，每首歌複製）
- 額外 3 個 leak 特徵（full only）

| Config | n_feat | R² | std |
|---|---:|---:|---:|
| **strict_song** | 1,165 | **0.290** | 0.009 |
| **full_song** | 1,168 | **0.317** | 0.005 |

### 13.5 為什麼 song-level R² 比 artist-level 低？

不是模型變爛，是**任務變嚴格**：

| 原因 | 細節 |
|---|---|
| Target std 大 6 倍 | song_hot std 0.234 vs artist_hot std 0.088 → R² 分母變大 |
| GroupKFold 嚴格 | 預測「完全沒見過的新藝人」，藝人層級 KFold 自動等價於 GroupKFold（每藝人一列），song-level 才有差異 |
| Within-artist 變異難抓 | 43 % 的 song hotness 變異是同藝人內的，需要每首歌不同特徵才能解釋 |

**變異數分解論證**：

- 純藝人層級資訊填均值 → R²(song) 上界 = `R²(artist) × ICC = 0.428 × 0.572 ≈ 0.244`
- 我們 strict_song R² = **0.290**
- **多出來的 0.046 是「per-track 特徵真正貢獻的 within-artist 訊號」**——印證 step 12 的猜測

### 13.6 步驟 14：song-level CatBoost + text features

`analysis/14_song_level_catboost.py`

把 step 13 的 HistGB + top-1000 multi-hot 換成 CatBoost + 全 7,538 tags 的 `text_features`，其餘設定（GroupKFold by artist_id、相同 audio + tl + year/duration + 3 個 leak 選項）不變。

| Config | HistGB top-1000 (step 13) | CatBoost text(7538) (step 14) | Δ |
|---|---:|---:|---:|
| strict_song | 0.290 | **0.286** | **−0.004** |
| full_song | 0.317 | **0.314** | **−0.003** |

**結果反直覺**：song-level + GroupKFold 下，CatBoost + 7,538 tags **沒有比** HistGB top-1000 更好，甚至略差。

#### 為什麼跟 artist-level 結果完全相反？

| 因素 | artist-level（step 11）| song-level + GroupKFold（step 14）|
|---|---|---|
| 「紅藝人有更多 tag」訊號 | 模型可從訓練資料記住此關聯 | GroupKFold 把訊號 neutralize 掉 |
| 稀有 tag 的價值 | 可作 artist 識別線索 | 對未見藝人沒幫助、反而是 overfit 風險 |
| Vocabulary 大小 | 30 → 7,538 帶 +0.07 | 1,000 → 7,538 幾乎不動 |

**重要修正**：之前在第 11–12 節說「CatBoost 的功勞 95 % 是 vocabulary」這句話需要打折——一旦改用 GroupKFold 對「沒見過的新藝人」公平評估，**vocabulary 的優勢大部分消失**。原本 artist-level 看到的 +0.07 提升，其中 **大半其實是「Echo Nest 給更紅藝人更多 tag」的隱性 leakage**。

### 13.7 結論補充

| 「對新藝人預測一首歌紅度」誠實天花板 | R² |
|---|---:|
| 純藝人層級填均值（理論上界，純 between）| ~0.244 |
| 加 per-track audio + tl + year/duration（HistGB top-1000）| **0.290** |
| 同上 + 換 CatBoost + 7,538 tags（**沒變好**）| 0.286 |
| 再加 leak 特徵（n_tracks 等） | 0.317 |

**0.05 的提升來自「audio + tl per track」**——比 step 12 在 artist-level 加同樣特徵的 +0.005 高 10 倍。確認**這些 track-level 特徵在 song level 才有意義**。

跨方法總結（不同任務的不同天花板）：

| 任務 | 預期 R² 上限 |
|---|---:|
| 預測藝人 hotness（strict + 30 tags） | 0.36 |
| 預測藝人 hotness（strict + 1000 tags） | 0.43 |
| 預測藝人 hotness（full + 7,538 tags + CatBoost，含 leakage）| 0.47 |
| **預測單首歌 hotness 對新藝人（strict + GroupKFold）** | **0.29**（HistGB top-1000 = CatBoost text）|
| **同上但允許 leak** | **0.32** |

### 13.8 步驟 15：song-level 特徵重要度

`analysis/15_song_level_importance.py`

對 step 13 的 HistGB 模型做 block-level + single-feature permutation importance（單一 GroupKFold split 的 80/20 hold-out）。

**Block 重要度**：

| Block | n features | ΔR² |
|---|---:|---:|
| **genre_top1000** | 1,000 | **0.224**（最大） |
| **per_track_yd**（year, duration, year_known） | 3 | **0.114** |
| artist_leak（n_tracks_a 等）| 3 | 0.096 |
| audio | 152 | 0.028 |
| tl | 10 | 0.004 |

**單一最強特徵**：`year_per_track` ΔR² = **0.112**，**比 152 個音訊整塊（0.028）大 4 倍**。

**Audio sub-family 拆解**：marsyas（72 cols）= 0.013（佔音訊半數），其餘 MFCC_simple、Area_MoM、lowlevel、LPC 各 ≤ 0.003。

**tl block 內部**：除 loudness（0.002）與 fade times（< 0.002）外，**mode、tempo、key、time_signature 對 hotness 預測幾乎沒貢獻**——這顛覆了「節奏 / 調性是音樂特徵」的直覺。

完整數值見 `results/15_song_level_importance/`：`block_importance.csv`、`single_feature_importance.csv`、`audio_subfamily_importance.csv`。

---

## 14. 最終結論

> **白話版**：根據用途不同，推薦不同的模型——「描述現有資料」可以用 R²=0.47 的 CatBoost 版本（含 leakage），但**真正部署到新藝人**要用 R²=0.43（CatBoost 誠實 + 全 tags）或 R²=0.24（線性誠實）的版本。**核心洞察是：藝人熱度由「曲風標籤、曝光、年代」三件事驅動，音訊本身只是次要訊號**——用 7,538 個原始 tag 比 152 個音訊特徵的訊息量還大。



| 場景 | 推薦模型 | R² |
|---|---|---:|
| **絕對最強 R²**（含 leakage） | **CatBoost full + 7,538 tags** | **0.47** |
| 描述 MSD 內藝人的 hotness | HistGB tuned on full 187 | 0.42 |
| **預測新藝人**（CatBoost + 全 tags） | **CatBoost strict + 7,538 tags** | **0.43** |
| 預測新藝人（top-30 tag 限制下） | HistGB tuned on strict 185 | 0.36 |
| **可解釋線性 baseline** | RidgeCV on strict 185（α=1） | **0.24** |
| 純音訊線性下界 | Ridge α=100 on 152 audio | 0.08 |
| 純音訊非線性上界 | HistGB tuned on 152 audio | 0.21 |

### Take-aways

1. **音訊單獨對藝人 hotness 的線性預測力極低（R² ≈ 0.08）**。共線性處理（VIF / PCA / Ridge）只能將其推到 0.084，差距無法靠線性方法跨越。
2. **5 個 metadata 特徵（year/duration/n_tracks/n_genres/year_known_ratio）的線性 R²（0.26）就遠超 152 個音訊特徵**。Hotness 主要被「曝光度 / 年代 / 標籤量」驅動，不是音訊本身。
3. **`n_tracks` 與 `n_genres` 是 popularity 的反向因果代理**——把它們從 187 維裡拿掉，HistGB R² 從 0.42 降到 0.36（−0.06），這 0.06 是「假」的預測力。
4. **誠實線性模型最終為 RidgeCV on strict 185 → R² = 0.24**，0.36 是同一資料 top-30 tag 限制下的非線性天花板。
5. 移除 leakage 後，**Area Method of Moments、Marsyas Chroma PeakRatio、Spectral Variability 是音訊中最有價值的子集合**——這也是線性係數重新洗牌時浮上來的特徵。
6. **真正能突破 R² 0.36 的不是換模型，而是把 genre 詞彙表從 30 開到 7,538**。CatBoost 的 `text_features` 把單一個 `term_norm` 欄位的重要度推到 32.7（超過全部 152 個音訊特徵總和 26.6），使 strict 0.36 → 0.43、full 0.42 → 0.47。**主要瓶頸是 feature engineering，不是 algorithm**。

---

## 15. 重現方式

```powershell
# 1. 從 SQLite 聚合到藝人層級
python analysis/01_aggregate_features.py    # ~22s

# 2. 共線性診斷 + 純音訊 baseline
python analysis/02_diagnostics_vif_pca.py   # ~13s

# 3. 擴展特徵集 B + D（HistGB CV 較慢）
python analysis/03_extended_baselines.py    # ~6 min

# 4. 特徵重要度
python analysis/04_feature_importance.py    # ~64s

# 5. Leakage check + 誠實 baseline
python analysis/05_leakage_check.py         # ~3 min

# 6. 記錄每個 configuration 用到的特徵集
python analysis/06_record_feature_sets.py   # <1s

# 7. 強化嚴謹度（iterative VIF / Lasso 掃描 / stratified CV）
python analysis/07_rigor_strengthening.py   # ~94s

# 8. CatBoost：用全 vocabulary 7,538 個 genre tags（需 pip install catboost）
python analysis/08_catboost.py              # ~16 min（4 個 5-fold CV + 1 final fit）

# 9. Vocabulary scaling（HistGB top-{30, 100, 300, 1000} 拆解 CatBoost 的貢獻）
python analysis/09_vocab_scaling.py         # ~7 min
```

### Song-level 延伸實驗（原先在 song-hotness 分支，已合回 main）

```powershell
# 10. 從 millionsongdataset.com 下載 H5（316 MB），抽取 song_hotttnesss 與 track-level 特徵
curl -L -o data/msd_summary_file.h5 "http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/msd_summary_file.h5"
python analysis/10_song_hotness_extract.py  # ~45s

# 11. 診斷 song_hotttnesss 內部變異
python analysis/11_song_vs_artist_hotness.py # <1s

# 12. Track-level Echo Nest 特徵加進 artist-level pipeline
python analysis/12_track_level_features.py  # ~8 min

# 13. Song-level pipeline + GroupKFold
python analysis/13_song_level_pipeline.py   # ~20 min

# 14. Song-level CatBoost + text features
python analysis/14_song_level_catboost.py   # ~18 min

# 15. Song-level feature importance (block + single perm on hold-out)
python analysis/15_song_level_importance.py # ~9 min
```

依賴：`numpy`, `pandas`, `scikit-learn`（建議 ≥ 1.3）。

所有腳本以 `data/MSD_with_all_features.db` 作為唯一原始資料來源；中間 cache 寫到 `analysis/cache/`，最終結果寫到 `analysis/results/`。

---

## 16. 已知限制

1. **`song_hotttnesss` 不在本 DB**——只能用 `artist_hotttnesss` 作為熱度代理；同藝人的不同歌存在差異（8.4 % 藝人，median spread 0.028），以 `AVG` 聚合時抹平了這個變異。
2. **`year_mean` 對 32 % 的藝人是 NaN**（所有歌 `year = 0`），用 median 補值，可能稀釋年代訊號。
3. **`term` 欄是 Echo Nest 演算法產生的 tag，不是經過策展的 genre**——含有國家／年代（"united states", "1990s"）等非曲風語義。
4. **Top 30 genre 是 hard cutoff**：其餘 7,514 個 tag 的訊息全部丟掉。
5. **`AVG()` 聚合假設藝人內所有歌權重相同**——對歌曲數差異很大的藝人（max 208 vs min 1）這不一定合理。
6. Echo Nest hotness 偏向 2010 年 US 主流市場，跨年代與跨地區可比性受限（從 `genre_united_states` 為負係數可窺一斑）。

---

## 17. 名詞解釋

| 術語 | 中文 | 白話解釋 |
|---|---|---|
| **R²** | 決定係數 | 模型預測能力的指標。0 = 跟亂猜一樣，1 = 完美預測，0.5 = 解釋了一半的變化。本報告所有 R² 都是 5-fold CV 下的平均。 |
| **5-fold CV** | 5 折交叉驗證 | 把資料切成 5 份，輪流用 4 份當訓練資料、1 份當測試，重複 5 次取平均。比單一切分更可靠地估計「對未見資料的預測力」。 |
| **線性模型** | 線性迴歸 | 最簡單的預測方式：把每個特徵乘上一個權重再加總，就是預測值。本報告中的 OLS、Ridge、Lasso 都屬此類。 |
| **OLS** | 普通最小平方法 | 最基本的線性迴歸，沒有任何規範化。共線性嚴重時係數會不穩定。 |
| **Ridge** | 嶺迴歸 | 線性迴歸 + 把所有係數「縮小」的懲罰。對付特徵間有重複資訊很有效。 |
| **Lasso** | 套索迴歸 | 線性迴歸 + 把不重要的係數「歸零」的懲罰。同時能做特徵選擇。 |
| **非線性模型** | — | 允許特徵間有複雜交互作用的模型。本報告用 HistGradientBoosting 和 RandomForest（兩者都是「樹」的集成）。 |
| **HistGradientBoosting** | 直方圖梯度提升 | 一種強力的非線性模型，本實驗的最佳預測者。能自動處理特徵交互作用與缺失值。 |
| **多重共線性** | — | 多個特徵在「量同一件事」，模型分不清是哪個特徵在起作用，係數會變得隨機。 |
| **VIF** | 變異膨脹因子 | 量化某特徵與其他特徵的「冗餘程度」。1 = 完全獨立，> 10 開始警戒，> 100 嚴重冗餘，10⁶ = 接近完全相依。 |
| **PCA** | 主成分分析 | 把多個相關特徵壓縮成「合成維度」（PC），保留最多訊息的少量維度。本資料 152 維壓到 30 個 PC 就保留 95 % 訊息。 |
| **條件數** | Condition Number | 矩陣「病態」程度的指標。10⁹ 表示計算精度只剩 6 位有效數字，OLS 係數會不穩定。 |
| **特徵** | Feature | 模型的「輸入欄位」。本實驗有 152 個音訊特徵 + 5 個基本欄位 + 30 個曲風指示。 |
| **標準化** | Standardization | 把不同單位的特徵縮放到同一範圍（z-score：平均 0、標準差 1），避免某個量級大的特徵主導模型。 |
| **特徵重要度** | Feature Importance | 量化每個特徵對預測的貢獻。 |
| **Permutation Importance** | 排列重要度 | 計算特徵重要度的方法：把某個特徵的值隨機打亂，看模型 R² 掉多少；掉越多 = 越重要。 |
| **資料洩漏** | Data Leakage | 用「結果相關」的特徵當輸入，模型看似神準但其實是「作弊」，對新案例無用。本實驗的 `n_tracks` / `n_genres` 就是典型例子。 |
| **Sentinel** | 哨兵值 | 用一個特殊值（例如 -1, 0）表示「資料缺失」。要記得過濾，否則會被當成真實值。 |
| **ICC** | 組內相關係數 | 介於 0 到 1。在「同藝人各歌共享 hotness」這類巢狀資料中，ICC = between-group variance / total variance。1 = 群組（藝人）完全決定數值；0 = 群組無解釋力。本資料 ICC ≈ 0.57，57 % 在藝人間、43 % 在藝人內。 |
| **GroupKFold** | 群組分折交叉驗證 | 比 KFold 更嚴格的 CV：把同一群組（這裡是同藝人）的所有列放進同一 fold，避免「模型在訓練時看過 Beyoncé 1 首歌、測試時看到她另一首」這種資訊洩漏。預測「未見過的新藝人」必用此法。 |
| **Echo Nest** | — | 一家音樂分析公司（後被 Spotify 收購），本資料的熱度分數和曲風標籤都來自他們的演算法。 |
| **MSD** | Million Song Dataset | 哥倫比亞大學釋出的 100 萬首歌音訊特徵資料集，本實驗的原始資料。 |
| **MFCC** | 梅爾頻率倒譜係數 | 一種描述音色（timbre）的特徵，是語音/音樂處理的標準工具。 |
| **LPC** | 線性預測係數 | 一種描述聲音「共振結構」的特徵，原本用於語音壓縮。 |
| **Chroma** | 音高類別 | 描述音樂的調性——12 個音名（A、A#、B…G#）各佔多少能量。 |
| **Method of Moments** | 矩量法 | 一種用「統計矩」（mean、variance、skewness…）描述頻譜形狀的特徵。 |
| **Marsyas** | — | 一個音樂訊號分析框架，本資料中的 124 個 marsyas 特徵就是用它計算的。 |
| **track_id / artist_id** | — | MSD 給每首歌、每位藝人的唯一識別碼，用來連接不同表格。 |

---

## 18. 檔案結構

```
.
├── README.md                       (本檔案)
├── load_msd.ipynb                  Jupyter 探索用 notebook
├── schema.md                       SQLite schema 文件 + Mermaid ER 圖
├── data/
│   ├── MSD_with_all_features.db
│   ├── flattened_MSD_with_all_features_remove_missing_values-001.csv
│   └── head200_flattened_MSD_with_all_features_remove_missing_values.csv
└── analysis/
    ├── 01_aggregate_features.py
    ├── 02_diagnostics_vif_pca.py
    ├── 03_extended_baselines.py
    ├── 04_feature_importance.py
    ├── 05_leakage_check.py
    ├── 06_record_feature_sets.py
    ├── 07_rigor_strengthening.py
    ├── 08_catboost.py
    ├── 09_vocab_scaling.py
    ├── 10_song_hotness_extract.py      ├── 11_song_vs_artist_hotness.py    ├── 12_track_level_features.py      ├── 13_song_level_pipeline.py       ├── 14_song_level_catboost.py       ├── 15_song_level_importance.py     ├── cache/
    │   ├── artist_audio_agg.pkl
    │   └── artist_extra.pkl
    └── results/                        每個子目錄對應一支腳本
        ├── 02_diagnostics_vif_pca/
        │   ├── vif.csv
        │   ├── pca_explained_variance.csv
        │   └── benchmark_audio_only.csv
        ├── 03_extended_baselines/
        │   └── benchmark_extended.csv
        ├── 04_feature_importance/
        │   ├── ridge_coefs_full.csv
        │   ├── perm_importance_full.csv
        │   └── importance_by_category.csv
        ├── 05_leakage_check/
        │   ├── benchmark_leakage.csv
        │   ├── ridge_coefs_strict.csv
        │   └── perm_importance_strict.csv
        ├── 06_record_feature_sets/
        │   ├── feature_sets_filtering.csv
        │   ├── feature_sets_summary.csv
        │   ├── feature_sets_long.csv
        │   └── feature_sets_wide.csv
        ├── 07_rigor_strengthening/
        │   ├── vif_iterative_trace.csv
        │   ├── vif_iterative_survivors.csv
        │   ├── lasso_active_sets.csv
        │   └── stratified_cv_benchmark.csv
        ├── 08_catboost/
        │   ├── benchmark_catboost.csv
        │   ├── feature_importance_catboost.csv
        │   └── feature_importance_catboost_by_category.csv
        ├── 09_vocab_scaling/
        │   └── benchmark_vocab_scaling.csv
        ├── 10_song_hotness_extract/        │   ├── coverage_report.csv
        │   └── join_with_artist_pipeline.csv
        ├── 11_song_vs_artist_hotness/        │   ├── diagnostics.csv
        │   ├── per_artist_stats.csv
        │   └── extreme_cases.csv
        ├── 12_track_level_features/        │   └── benchmark_track_level.csv
        ├── 13_song_level_pipeline/         │   └── benchmark_song_level.csv
        ├── 14_song_level_catboost/         │   └── benchmark_song_catboost.csv
        └── 15_song_level_importance/             ├── block_importance.csv
            ├── single_feature_importance.csv
            └── audio_subfamily_importance.csv
```
