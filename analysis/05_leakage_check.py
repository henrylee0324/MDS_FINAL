"""
Step 05: Leakage check via progressive feature ablation.

`n_tracks` and `n_genres` are popularity proxies (an artist with more
tracks/tags in MSD is, by construction, an artist someone bothered to
catalogue extensively). Including them in the feature set inflates R^2
beyond what's achievable for an unseen artist. We strip them out and
report the honest ceiling.

Configurations (all reuse the audio + selected numeric + genre stack):

  full                      [year_mean, duration_mean, year_known_ratio, n_tracks, n_genres]
  drop n_tracks             [year_mean, duration_mean, year_known_ratio, n_genres]
  drop n_genres             [year_mean, duration_mean, year_known_ratio, n_tracks]
  drop both leak features   [year_mean, duration_mean, year_known_ratio]
  + drop year_known_ratio   [year_mean, duration_mean]
  audio + genre only        []
  audio only                no genre, no numeric

For each, 5-fold CV R^2 is reported for both Ridge and HistGB.
On the strict set (without n_tracks + n_genres) we also run permutation
importance to see what features carry the load.

Outputs to results/:
  benchmark_leakage.csv         R^2 for every (model, config) pair
  ridge_coefs_strict.csv        Ridge coefficients on the strict set
  perm_importance_strict.csv    HistGB permutation importance on the strict set
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, train_test_split

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results'

def categorize(name, num_set):
    if name in num_set:                     return 'numeric'
    if name.startswith('genre_'):           return 'genre'
    if name.startswith('Area_Method'):      return 'audio:Area_MoM'
    if name.startswith('LPC_'):             return 'audio:LPC'
    if name.startswith('MFCC_'):            return 'audio:MFCC_simple'
    if name.startswith(('Spectral_','Compactness','Root_Mean','Fraction','Zero_')):
        return 'audio:lowlevel'
    if 'Mem20' in name:                     return 'audio:marsyas'
    return 'audio:other'

def main():
    t0 = time.time()
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    extra = pd.read_pickle(CACHE / 'artist_extra.pkl')

    audio_feats = [c for c in audio.columns if c not in ('artist_id','hotness','n_tracks')]
    mom_prune    = [c for c in audio_feats if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_feats if 'Mem20_MFCC' in c]
    audio_keep   = [c for c in audio_feats if c not in set(mom_prune)|set(marsyas_mfcc)]
    clean = audio.dropna(subset=audio_feats).reset_index(drop=True)
    df = clean.merge(extra, on='artist_id', how='left')
    Xa = df[audio_keep].values.astype(np.float64)
    audio_keep = [k for k, kp in zip(audio_keep, Xa.std(axis=0)>0) if kp]
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    y = df['hotness'].values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def stack(num_used, include_audio=True, include_genre=True):
        blocks, names = [], []
        if include_audio:
            blocks.append(df[audio_keep].values.astype(np.float64))
            names += audio_keep
        if num_used:
            blocks.append(df[num_used].values.astype(np.float64))
            names += num_used
        if include_genre:
            blocks.append(df[genre_cols].values.astype(np.float64))
            names += genre_cols
        return np.hstack(blocks), names

    def cv_ridge(X, names):
        # stack puts non-genre cols first, then genre; standardize the non-genre block
        n_to_scale = sum(1 for n in names if not n.startswith('genre_'))
        Xs = X.copy()
        if n_to_scale > 0:
            Xs[:, :n_to_scale] = SimpleImputer(strategy='median').fit_transform(Xs[:, :n_to_scale])
            Xs[:, :n_to_scale] = StandardScaler().fit_transform(Xs[:, :n_to_scale])
        return cross_val_score(RidgeCV(alphas=[0.1,1,10,100,1000]),
                                Xs, y, cv=cv, scoring='r2', n_jobs=-1).mean()

    def cv_hgb(X):
        hgb = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.03, max_depth=8, l2_regularization=0.1,
            min_samples_leaf=20, random_state=42)
        return cross_val_score(hgb, X, y, cv=cv, scoring='r2', n_jobs=-1).mean()

    print('=== Progressive leakage ablation (5-fold CV R^2) ===')
    configs = [
        ('full',                            ['year_mean','duration_mean','year_known_ratio','n_tracks','n_genres'], True, True),
        ('drop n_tracks',                   ['year_mean','duration_mean','year_known_ratio','n_genres'],            True, True),
        ('drop n_genres',                   ['year_mean','duration_mean','year_known_ratio','n_tracks'],            True, True),
        ('drop n_tracks + n_genres',        ['year_mean','duration_mean','year_known_ratio'],                       True, True),
        ('+ drop year_known_ratio',         ['year_mean','duration_mean'],                                          True, True),
        ('audio + genre only',              [],                                                                     True, True),
        ('audio only',                      [],                                                                     True, False),
    ]
    rows = []
    for name, num, ia, ig in configs:
        X, names = stack(num, include_audio=ia, include_genre=ig)
        r = cv_ridge(X, names) if (ia or num or ig) else float('nan')
        h = cv_hgb(X)
        rows.append({'config': name, 'n_feat': X.shape[1], 'ridge_r2': r, 'hgb_r2': h})
        print(f'  {name:35s}  Ridge={r:.4f}  HistGB={h:.4f}  ({X.shape[1]} feats)')
    pd.DataFrame(rows).to_csv(RESULT / 'benchmark_leakage.csv', index=False)

    # ---- On the strict set: Ridge coefs + HistGB permutation importance ----
    print('\n=== Strict set deep-dive (drop n_tracks + n_genres) ===')
    strict_num = ['year_mean','duration_mean','year_known_ratio']
    X_strict, strict_names = stack(strict_num, include_audio=True, include_genre=True)
    X_imp = SimpleImputer(strategy='median').fit_transform(X_strict)

    n_aud_num = len(audio_keep) + len(strict_num)
    X_ridge = X_imp.copy()
    X_ridge[:, :n_aud_num] = StandardScaler().fit_transform(X_ridge[:, :n_aud_num])
    ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
    ridge.fit(X_ridge, y)
    coefs = pd.DataFrame({'feature': strict_names, 'coef': ridge.coef_})
    coefs['abs_coef'] = coefs['coef'].abs()
    coefs.sort_values('abs_coef', ascending=False).to_csv(RESULT / 'ridge_coefs_strict.csv', index=False)
    print(f'Strict Ridge: alpha={ridge.alpha_}  train R2={ridge.score(X_ridge, y):.4f}')
    print('Top 10 positive coefficients:')
    for _, r in coefs.sort_values('coef', ascending=False).head(10).iterrows():
        print(f'  {r["coef"]:+.4f}   {r["feature"]}')

    X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2, random_state=42)
    hgb = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.03, max_depth=8, l2_regularization=0.1,
        min_samples_leaf=20, random_state=42)
    hgb.fit(X_tr, y_tr)
    print(f'\nStrict HistGB hold-out R2 = {hgb.score(X_te, y_te):.4f}')
    perm = permutation_importance(hgb, X_te, y_te, n_repeats=5,
                                   random_state=42, n_jobs=-1, scoring='r2')
    imp_df = pd.DataFrame({
        'feature':  strict_names,
        'imp_mean': perm.importances_mean,
        'imp_std':  perm.importances_std,
    })
    imp_df['category'] = imp_df['feature'].apply(lambda n: categorize(n, set(strict_num)))
    imp_df.sort_values('imp_mean', ascending=False).to_csv(
        RESULT / 'perm_importance_strict.csv', index=False)
    print('Top 15 by permutation importance:')
    for _, r in imp_df.sort_values('imp_mean', ascending=False).head(15).iterrows():
        print(f'  {r["imp_mean"]:+.5f}   {r["feature"]}')
    cat_imp = (imp_df.groupby('category')['imp_mean']
               .agg(['sum','count']).sort_values('sum', ascending=False))
    print('\nCategory totals on strict set:')
    print(cat_imp.to_string())

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
