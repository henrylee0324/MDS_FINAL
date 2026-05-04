"""
Step 09: Genre vocabulary scaling on HistGB.

CatBoost reached strict R^2 = 0.430 (vs HistGB 0.357) when given the raw
`term` text feature with 7,538 unique tags. The question this script
answers: how much of that +0.073 gap was the vocabulary expansion alone,
versus CatBoost's internal text-handling magic?

Approach: feed HistGB the same artist matrix but vary the genre vocabulary
from top-30 (current baseline) to top-1000 multi-hot indicators. If HistGB
catches up to CatBoost's 0.430 it's purely a vocabulary effect; if it
plateaus below, CatBoost's BoW representation contributes additional
signal beyond raw counts.

Configurations (5-fold CV R^2):

  strict + top-N multi-hot, for N in {30, 100, 300, 1000}
  full   + top-N multi-hot, for the same Ns

Outputs:
  results/09_vocab_scaling/benchmark_vocab_scaling.csv
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

ROOT   = Path(__file__).resolve().parent.parent
DB     = ROOT / 'data' / 'MSD_with_all_features.db'
HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results' / '09_vocab_scaling'
RESULT.mkdir(exist_ok=True, parents=True)

NUMERIC_FULL   = ['year_mean','duration_mean','year_known_ratio','n_tracks','n_genres']
NUMERIC_STRICT = ['year_mean','duration_mean','year_known_ratio']

def main():
    t0 = time.time()
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    extra = pd.read_pickle(CACHE / 'artist_extra.pkl')

    print('Re-fetching raw term column from SQLite...')
    with sqlite3.connect(DB) as conn:
        terms = pd.read_sql("""
            SELECT artist_id, MAX(term) AS term
            FROM merged_partition1
            WHERE CAST(artist_hotttnesss AS REAL) > 0
            GROUP BY artist_id
        """, conn)

    audio_feats = [c for c in audio.columns if c not in ('artist_id','hotness','n_tracks')]
    mom_prune    = [c for c in audio_feats if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_feats if 'Mem20_MFCC' in c]
    audio_keep   = [c for c in audio_feats if c not in set(mom_prune)|set(marsyas_mfcc)]
    clean = audio.dropna(subset=audio_feats).reset_index(drop=True)
    Xa = clean[audio_keep].values.astype(np.float64)
    audio_keep = [k for k, kp in zip(audio_keep, Xa.std(axis=0)>0) if kp]

    df = clean.merge(extra, on='artist_id').merge(terms, on='artist_id')
    df['term_list'] = df['term'].fillna('').apply(
        lambda s: [t.strip() for t in s.split(',') if t.strip()])

    counter = Counter()
    for lst in df['term_list']:
        counter.update(lst)
    total_unique = len(counter)
    print(f'Joined: {df.shape}    unique tags: {total_unique}')

    # Pre-compute set membership for fast multi-hot
    df['term_set'] = df['term_list'].apply(set)

    y = df['hotness'].values
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def build_multihot(top_n):
        """Return a (n_artists, top_n) float32 indicator matrix."""
        if top_n == 0:
            return np.zeros((len(df), 0), dtype=np.float32)
        top_tags = [t for t, _ in counter.most_common(top_n)]
        M = np.zeros((len(df), top_n), dtype=np.float32)
        for j, tag in enumerate(top_tags):
            M[:, j] = df['term_set'].apply(lambda s: tag in s).values
        return M, top_tags

    def cv_r2_hgb(X):
        hgb = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.03, max_depth=8,
            l2_regularization=0.1, min_samples_leaf=20, random_state=42)
        return cross_val_score(hgb, X, y, cv=cv, scoring='r2', n_jobs=-1)

    rows = []
    for N in [30, 100, 300, 1000]:
        print(f'\n=== top-{N} tags ({time.time()-t0:.1f}s) ===')
        M, tags = build_multihot(N)

        # strict
        X_strict = np.hstack([
            df[audio_keep].values.astype(np.float32),
            df[NUMERIC_STRICT].values.astype(np.float32),
            M
        ])
        scores = cv_r2_hgb(X_strict)
        rows.append({'config':'strict', 'top_n':N,
                     'n_feat':X_strict.shape[1],
                     'cv_r2':scores.mean(), 'cv_std':scores.std()})
        print(f'  strict + top-{N:>4d}:  R^2 = {scores.mean():.4f} +/- {scores.std():.4f}'
              f'   ({X_strict.shape[1]} feats)')

        # full
        X_full = np.hstack([
            df[audio_keep].values.astype(np.float32),
            df[NUMERIC_FULL].values.astype(np.float32),
            M
        ])
        scores = cv_r2_hgb(X_full)
        rows.append({'config':'full', 'top_n':N,
                     'n_feat':X_full.shape[1],
                     'cv_r2':scores.mean(), 'cv_std':scores.std()})
        print(f'  full   + top-{N:>4d}:  R^2 = {scores.mean():.4f} +/- {scores.std():.4f}'
              f'   ({X_full.shape[1]} feats)')

    bench = pd.DataFrame(rows)
    bench.to_csv(RESULT / 'benchmark_vocab_scaling.csv', index=False)
    print(f'\n=== Saved -> {RESULT/"benchmark_vocab_scaling.csv"} ===')
    print(bench.to_string(index=False))

    print('\n=== Reference points ===')
    print('  HistGB strict + top-30:  R^2 = 0.357 (step 03/05)')
    print('  HistGB full   + top-30:  R^2 = 0.422 (step 03)')
    print('  CatBoost strict + text(7538):  R^2 = 0.430 (step 08)')
    print('  CatBoost full   + text(7538):  R^2 = 0.470 (step 08)')

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
