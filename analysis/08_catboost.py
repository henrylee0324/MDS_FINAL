"""
Step 08: CatBoost benchmark - test whether full-vocabulary genre tags help.

The genre vocabulary in `term` has 7,544 unique tags but only the top 30 are
encoded as multi-hot in the existing pipeline (HistGB / Ridge cannot
practically eat all 7,544 as columns). CatBoost's text_features parameter
tokenizes the raw comma-separated string and produces internal numeric
representations, so it can use the entire vocabulary without column
explosion.

Four configurations (5-fold CV R^2):

  catboost_strict_185        audio + 3 numeric + 30 genre multi-hot
                             apples-to-apples vs HistGB strict (R^2 = 0.357)

  catboost_strict_text       audio + 3 numeric + raw `term` as text feature
                             same scope but with ALL 7,544 tags via CatBoost

  catboost_full_187          audio + 5 numeric + 30 genre multi-hot
                             apples-to-apples vs HistGB full (R^2 = 0.422)

  catboost_full_text         audio + 5 numeric + raw `term` as text feature

Outputs:
  results/08_catboost/benchmark_catboost.csv
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

ROOT   = Path(__file__).resolve().parent.parent
DB     = ROOT / 'data' / 'MSD_with_all_features.db'
HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results' / '08_catboost'
RESULT.mkdir(exist_ok=True, parents=True)

NUMERIC_FULL   = ['year_mean','duration_mean','year_known_ratio','n_tracks','n_genres']
NUMERIC_STRICT = ['year_mean','duration_mean','year_known_ratio']

def normalize_term(s):
    """Comma-separated tags -> whitespace-separated, multi-word tags joined by '_'."""
    if pd.isna(s) or not s:
        return ''
    return ' '.join(t.strip().replace(' ', '_') for t in s.split(',') if t.strip())

def main():
    t0 = time.time()
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    extra = pd.read_pickle(CACHE / 'artist_extra.pkl')

    # Re-fetch raw `term` per artist (cache dropped it after extracting top-30)
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

    df = clean.merge(extra, on='artist_id', how='left').merge(terms, on='artist_id', how='left')
    df['term_norm'] = df['term'].apply(normalize_term)
    n_unique_tags = len(set(t for s in df['term_norm'] for t in s.split() if t))
    print(f'Joined: {df.shape}    unique tags in term: {n_unique_tags}')
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    y = df['hotness'].values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def cv_r2(features, text_features=None):
        # Manual CV because CatBoost's sklearn shim breaks clone() when
        # text_features is set, so cross_val_score errors out.
        X = df[features].copy()
        text_set = set(text_features or [])
        for c in features:
            if c not in text_set and X[c].dtype.kind in 'fc':
                X[c] = X[c].fillna(X[c].median())
        cb_args = dict(iterations=600, learning_rate=0.05, depth=8, l2_leaf_reg=3,
                       random_seed=42, verbose=0, thread_count=-1,
                       allow_writing_files=False)
        if text_features:
            cb_args['text_features'] = text_features
        scores = []
        for tr, te in cv.split(X):
            model = CatBoostRegressor(**cb_args)
            model.fit(X.iloc[tr], y[tr])
            scores.append(r2_score(y[te], model.predict(X.iloc[te])))
        return float(np.mean(scores)), float(np.std(scores))

    rows = []

    print(f'\n=== Running 4 CatBoost configurations ({time.time()-t0:.1f}s) ===')

    feats_strict_top30 = audio_keep + NUMERIC_STRICT + genre_cols
    m, s = cv_r2(feats_strict_top30)
    rows.append({'config':'strict_185_top30genre', 'n_feat':len(feats_strict_top30),
                 'tag_vocab':30, 'cv_r2':m, 'cv_std':s})
    print(f'  strict, 30 genre multi-hot:        R2 = {m:.4f} +/- {s:.4f}   ({len(feats_strict_top30)} feats)   ({time.time()-t0:.1f}s)')

    feats_strict_text = audio_keep + NUMERIC_STRICT + ['term_norm']
    m, s = cv_r2(feats_strict_text, text_features=['term_norm'])
    rows.append({'config':'strict_term_text', 'n_feat':len(feats_strict_text),
                 'tag_vocab':n_unique_tags, 'cv_r2':m, 'cv_std':s})
    print(f'  strict, term as text (all tags):   R2 = {m:.4f} +/- {s:.4f}   ({len(feats_strict_text)} cols, {n_unique_tags} tags via text)   ({time.time()-t0:.1f}s)')

    feats_full_top30 = audio_keep + NUMERIC_FULL + genre_cols
    m, s = cv_r2(feats_full_top30)
    rows.append({'config':'full_187_top30genre', 'n_feat':len(feats_full_top30),
                 'tag_vocab':30, 'cv_r2':m, 'cv_std':s})
    print(f'  full, 30 genre multi-hot:          R2 = {m:.4f} +/- {s:.4f}   ({len(feats_full_top30)} feats)   ({time.time()-t0:.1f}s)')

    feats_full_text = audio_keep + NUMERIC_FULL + ['term_norm']
    m, s = cv_r2(feats_full_text, text_features=['term_norm'])
    rows.append({'config':'full_term_text', 'n_feat':len(feats_full_text),
                 'tag_vocab':n_unique_tags, 'cv_r2':m, 'cv_std':s})
    print(f'  full, term as text (all tags):     R2 = {m:.4f} +/- {s:.4f}   ({len(feats_full_text)} cols, {n_unique_tags} tags via text)   ({time.time()-t0:.1f}s)')

    bench = pd.DataFrame(rows)
    bench.to_csv(RESULT / 'benchmark_catboost.csv', index=False)
    print(f'\n=== Saved -> {RESULT/"benchmark_catboost.csv"} ===')
    print(bench.to_string(index=False))

    # Reference points for comparison (from prior steps)
    print('\n=== Reference (from previous runs) ===')
    print('  HistGB strict_185:  R2 = 0.3569 +/- 0.0039  (step 03)')
    print('  HistGB full_187:    R2 = 0.4217 +/- 0.0018  (step 03)')

    # --- Feature importance on the best config (full + text) ---
    print(f'\n=== Training final CatBoost (full + text) for feature importance ({time.time()-t0:.1f}s) ===')
    final_feats = audio_keep + NUMERIC_FULL + ['term_norm']
    X_final = df[final_feats].copy()
    for c in final_feats:
        if c != 'term_norm' and X_final[c].dtype.kind in 'fc':
            X_final[c] = X_final[c].fillna(X_final[c].median())
    final_model = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=8, l2_leaf_reg=3,
        text_features=['term_norm'], random_seed=42, verbose=0,
        thread_count=-1, allow_writing_files=False,
    )
    final_model.fit(X_final, y)

    def categorize(name):
        if name in NUMERIC_FULL:                return 'numeric'
        if name == 'term_norm':                 return 'genre_text'
        if name.startswith('Area_Method'):      return 'audio:Area_MoM'
        if name.startswith('LPC_'):             return 'audio:LPC'
        if name.startswith('MFCC_'):            return 'audio:MFCC_simple'
        if name.startswith(('Spectral_','Compactness','Root_Mean','Fraction','Zero_')):
            return 'audio:lowlevel'
        if 'Mem20' in name:                     return 'audio:marsyas'
        return 'audio:other'

    imp_df = pd.DataFrame({
        'feature':    final_feats,
        'importance': final_model.feature_importances_,
    })
    imp_df['category'] = imp_df['feature'].apply(categorize)
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    imp_df.to_csv(RESULT / 'feature_importance_catboost.csv', index=False)

    cat_imp = imp_df.groupby('category')['importance'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    cat_imp.to_csv(RESULT / 'feature_importance_catboost_by_category.csv')

    print('\nTop 15 features by CatBoost importance:')
    for _, r in imp_df.head(15).iterrows():
        print(f'  {r["importance"]:>7.3f}   [{r["category"]:18s}] {r["feature"]}')
    print('\nCategory totals:')
    print(cat_imp.to_string())

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
