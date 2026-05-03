# MSD SQLite Schema

DB: `data/MSD_with_all_features.db` — 8 tables, no declared foreign keys. All inter-table links go through `track_id`.

> **Render this file**: open in VS Code with the *Markdown Preview Mermaid Support* extension, or paste the diagram block into <https://mermaid.live>.

## Logical ER (normalized — what you should mentally model)

```mermaid
erDiagram
    SONGS ||--o| AREA_OF_MOMENTS           : track_id
    SONGS ||--o| LINEAR_PREDICTIVE_CODING  : track_id
    SONGS ||--o| LOW_LEVEL_FEATURES        : track_id
    SONGS ||--o| MFCC_FEATURES             : track_id
    SONGS ||--o| METHOD_OF_MOMENTS         : track_id
    SONGS ||--o| MARSYAS_TIMBRAL_FEATURES  : track_id

    SONGS {
        TEXT track_id PK "1,000,000 rows"
        TEXT title
        TEXT song_id
        TEXT release
        TEXT artist_id
        TEXT artist_mbid
        TEXT artist_name
        TEXT duration "stored as TEXT, cast to float"
        TEXT artist_familiarity "stored as TEXT, cast to float"
        TEXT artist_hotttnesss "stored as TEXT, cast to float"
        TEXT year "stored as TEXT, 0 = unknown (~48%)"
        TEXT track_7digitalid
        TEXT shs_perf "performance ID, -1 = none (~98.5%)"
        TEXT shs_work "work ID, 0 = none (~98%)"
        TEXT term "comma-joined genre tags"
        TEXT similar "comma-joined similar artist_ids"
    }

    AREA_OF_MOMENTS {
        REAL Area_MoM_Std_1_to_10 "10 cols"
        REAL Area_MoM_Avg_1_to_10 "10 cols"
        TEXT track_id FK "994,604 rows"
    }

    LINEAR_PREDICTIVE_CODING {
        REAL LPC_Std_1_to_10 "10 cols"
        REAL LPC_Avg_1_to_10 "10 cols"
        TEXT track_id FK "994,623 rows"
    }

    LOW_LEVEL_FEATURES {
        REAL Spectral_Centroid_Std_and_Avg
        REAL Spectral_Rolloff_Point_Std_and_Avg
        REAL Spectral_Flux_Std_and_Avg
        REAL Compactness_Std_and_Avg
        REAL Spectral_Variability_Std_and_Avg
        REAL Root_Mean_Square_Std_and_Avg
        REAL Fraction_Of_Low_Energy_Windows_Std_and_Avg
        REAL Zero_Crossings_Std_and_Avg
        TEXT track_id FK "994,623 rows; 16 feature cols"
    }

    MFCC_FEATURES {
        REAL MFCC_Std_1_to_13 "13 cols"
        REAL MFCC_Avg_1_to_13 "13 cols"
        TEXT track_id FK "994,623 rows"
    }

    METHOD_OF_MOMENTS {
        REAL MoM_Std_1_to_5 "5 cols"
        REAL MoM_Avg_1_to_5 "5 cols"
        TEXT track_id FK "994,623 rows"
    }

    MARSYAS_TIMBRAL_FEATURES {
        REAL Mean_Acc5_Mean_Mem20_x31 "31 cols"
        REAL Mean_Acc5_Std_Mem20_x31 "31 cols"
        REAL Std_Acc5_Mean_Mem20_x31 "31 cols"
        REAL Std_Acc5_Std_Mem20_x31 "31 cols"
        TEXT track_id FK "995,000 rows; 124 feature cols"
    }
```

## How `MARSYAS_TIMBRAL_FEATURES` 124 cols are organized

`{outer}_Acc5_{inner}_Mem20_<feature>_..._AudioCh0`, where outer/inner ∈ {Mean, Std} (4 combos) × 31 features.

```mermaid
flowchart LR
    A[outer aggregator<br/>Mean / Std] --> B[Acc5]
    B --> C[inner aggregator<br/>Mean / Std]
    C --> D[Mem20]
    D --> E[31 base features]
    E --> E1[ZeroCrossings]
    E --> E2[Centroid / Rolloff / Flux]
    E --> E3[MFCC0..MFCC12<br/>13 cols]
    E --> E4[PeakRatio_Chroma_A..G#<br/>12 cols]
    E --> E5[PeakRatio_Average_Chroma_A]
    E --> E6[PeakRatio_Minimum_Chroma_A]

    style A fill:#eef
    style C fill:#eef
    style E fill:#fee
```

Total = 4 × 31 = **124** feature columns + `track_id` = 125.

## The denormalized table (separate from the ER above)

`merged_partition1` is a **flat join** of all 7 tables: 16 metadata + 216 audio feature columns = **232 columns × 1,000,000 rows**. It is **not** a normalized entity — it duplicates everything you can already get by joining the others on `track_id`. Loading the full table costs ~2 GB of RAM.

```mermaid
flowchart TB
    S[SONGS<br/>16 metadata cols] --> M
    A[AREA_OF_MOMENTS<br/>20 cols] --> M
    L[LINEAR_PREDICTIVE_CODING<br/>20 cols] --> M
    LL[LOW_LEVEL_FEATURES<br/>16 cols] --> M
    MA[MARSYAS_TIMBRAL_FEATURES<br/>124 cols] --> M
    MF[MFCC_FEATURES<br/>26 cols] --> M
    MM[METHOD_OF_MOMENTS<br/>10 cols] --> M
    M[merged_partition1<br/>232 cols × 1M rows<br/>~2 GB in memory]

    style M fill:#fee
```

## Quick reference: cardinalities

| Table | Rows | "Missing" tracks vs songs |
|---|---:|---:|
| `songs` | 1,000,000 | — (canonical track set) |
| `marsyas_timbral_features` | 995,000 | 5,000 |
| `linear_predictive_coding` | 994,623 | 5,377 |
| `low_level_features` | 994,623 | 5,377 |
| `mfcc_features` | 994,623 | 5,377 |
| `method_of_moments` | 994,623 | 5,377 |
| `area_of_moments` | 994,604 | 5,396 |
| `merged_partition1` | 1,000,000 | 0 (NULLs filled where features absent) |

⚠️ The 5k–5.4k rows missing from each feature table do **not** fully overlap with `songs` — an `INNER JOIN` across all six on `track_id` gives slightly fewer than 994,604 rows.
