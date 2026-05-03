"""
Step 03: Extended-feature baselines (B + D).

Combines audio (152) + numeric (5: year_mean, duration_mean, year_known_ratio,
n_tracks, n_genres) + genre top-30 multi-hot = 187 features. Runs:

  B (Ridge linear)  : RidgeCV alphas=[0.1, 1, 10, 100, 1000]
                      across feature subsets (audio only, +numeric, +genre, all,
                      non-audio only, numeric only, genre only)

  D (Nonlinear)     : HistGradientBoosting (default + tuned), RandomForest

For Ridge, the numeric/audio blocks are median-imputed and standardized; genre
indicators are passed through as 0/1. Tree models receive raw (NaN-tolerant for
HistGB; median-imputed for RF).

Outputs to results/:
  benchmark_extended.csv    one row per (model, feature subset)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results' / '03_extended_baselines'
RESULT.mkdir(exist_ok=True, parents=True)

def main():
    t0 = time.time()
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    extra = pd.read_pickle(CACHE / 'artist_extra.pkl')

    audio_feats = [c for c in audio.columns if c not in ('artist_id','hotness','n_tracks')]
    mom_prune    = [c for c in audio_feats if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_feats if 'Mem20_MFCC' in c]
    audio_keep = [c for c in audio_feats if c not in set(mom_prune)|set(marsyas_mfcc)]

    clean = audio.dropna(subset=audio_feats).reset_index(drop=True)
    df = clean.merge(extra, on='artist_id', how='left')

    X_aud_raw = df[audio_keep].values.astype(np.float64)
    audio_keep = [k for k, kp in zip(audio_keep, X_aud_raw.std(axis=0)>0) if kp]
    print(f'Features: {len(audio_keep)} audio + 5 numeric + 30 genre')

    genre_cols    = [c for c in df.columns if c.startswith('genre_')]
    numeric_extra = ['year_mean','duration_mean','year_known_ratio','n_tracks','n_genres']
    y = df['hotness'].values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def build(use_audio, use_num, use_gen):
        blocks, parts = [], []
        pos = 0
        if use_audio:
            blocks.append(df[audio_keep].values)
            n = len(audio_keep)
            parts.append(('aud', Pipeline([('imp', SimpleImputer(strategy='median')),
                                           ('sc',  StandardScaler())]),
                          list(range(pos, pos+n)))); pos += n
        if use_num:
            blocks.append(df[numeric_extra].values)
            n = len(numeric_extra)
            parts.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                           ('sc',  StandardScaler())]),
                          list(range(pos, pos+n)))); pos += n
        if use_gen:
            blocks.append(df[genre_cols].values.astype(np.float64))
            n = len(genre_cols)
            parts.append(('gen', 'passthrough', list(range(pos, pos+n)))); pos += n
        X = np.hstack(blocks) if blocks else np.empty((len(df), 0))
        return X, ColumnTransformer(parts)

    def cv_ridge(X, ct):
        pipe = Pipeline([('ct', ct), ('reg', RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))])
        return cross_val_score(pipe, X, y, cv=cv, scoring='r2', n_jobs=-1).mean()

    def cv_tree(X, model):
        return cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1).mean()

    rows = []
    print('\n=== B: Ridge (5-fold CV) ===')
    for name, ua, un, ug in [
        ('audio only',                  True,  False, False),
        ('audio + numeric',             True,  True,  False),
        ('audio + genre',               True,  False, True),
        ('audio + numeric + genre',     True,  True,  True),
        ('non-audio only (num+genre)',  False, True,  True),
        ('numeric only',                False, True,  False),
        ('genre only',                  False, False, True),
    ]:
        X, ct = build(ua, un, ug)
        r = cv_ridge(X, ct)
        rows.append({'model': 'Ridge', 'subset': name, 'n_feat': X.shape[1], 'cv_r2': r})
        print(f'  {name:30s}  R2 = {r:.4f}   ({X.shape[1]} feats)')

    print(f'\n=== D: Nonlinear models ({time.time()-t0:.1f}s) ===')
    X_full, _ = build(True, True, True)
    hgb_default = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    hgb_tuned = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.03, max_depth=8, l2_regularization=0.1,
        min_samples_leaf=20, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)

    r = cv_tree(X_full, hgb_default)
    rows.append({'model':'HistGB default','subset':'audio + numeric + genre','n_feat':X_full.shape[1],'cv_r2':r})
    print(f'  HistGB default                R2 = {r:.4f}')

    r = cv_tree(X_full, hgb_tuned)
    rows.append({'model':'HistGB tuned','subset':'audio + numeric + genre','n_feat':X_full.shape[1],'cv_r2':r})
    print(f'  HistGB tuned                  R2 = {r:.4f}')

    r = cv_tree(df[audio_keep].values, hgb_tuned)
    rows.append({'model':'HistGB tuned','subset':'audio only','n_feat':len(audio_keep),'cv_r2':r})
    print(f'  HistGB tuned (audio only)     R2 = {r:.4f}')

    X_imp = SimpleImputer(strategy='median').fit_transform(X_full)
    r = cv_tree(X_imp, rf)
    rows.append({'model':'RandomForest','subset':'audio + numeric + genre','n_feat':X_full.shape[1],'cv_r2':r})
    print(f'  RandomForest                  R2 = {r:.4f}')

    pd.DataFrame(rows).to_csv(RESULT / 'benchmark_extended.csv', index=False)
    print(f'\nDone in {time.time()-t0:.1f}s. -> {RESULT/"benchmark_extended.csv"}')

if __name__ == '__main__':
    main()
