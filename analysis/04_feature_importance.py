"""
Step 04: Feature importance on the full feature set (187 features).

Two complementary views:

  * Ridge standardized coefficients
      RidgeCV alphas=[0.1, 1, 10, 100, 1000].
      Audio + numeric blocks median-imputed and z-scored; genre passthrough.
      Coefficients are directly comparable in standardized units.

  * HistGradientBoosting permutation importance
      Tuned model (max_iter=600, lr=0.03, max_depth=8, l2=0.1, min_samples_leaf=20).
      Train/test 80/20 split; importance computed on the held-out 20%.
      n_repeats=5.

Outputs to results/:
  ridge_coefs_full.csv         all 187 coefficients (sorted by abs value)
  perm_importance_full.csv     all 187 features with mean & std importance
  importance_by_category.csv   aggregate permutation importance by feature family
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
from sklearn.model_selection import train_test_split

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results' / '04_feature_importance'
RESULT.mkdir(exist_ok=True, parents=True)

NUMERIC = ['year_mean','duration_mean','year_known_ratio','n_tracks','n_genres']

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
    all_feats = audio_keep + NUMERIC + genre_cols

    X_raw = df[all_feats].values.astype(np.float64)
    y     = df['hotness'].values

    X_imp = SimpleImputer(strategy='median').fit_transform(X_raw)

    # ---- Ridge ----
    n_aud_num = len(audio_keep) + len(NUMERIC)
    X_ridge = X_imp.copy()
    X_ridge[:, :n_aud_num] = StandardScaler().fit_transform(X_ridge[:, :n_aud_num])
    ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
    ridge.fit(X_ridge, y)
    print(f'RidgeCV alpha={ridge.alpha_}   train R2={ridge.score(X_ridge, y):.4f}')
    coefs = pd.DataFrame({'feature': all_feats, 'coef': ridge.coef_})
    coefs['abs_coef'] = coefs['coef'].abs()
    coefs.sort_values('abs_coef', ascending=False).to_csv(RESULT / 'ridge_coefs_full.csv', index=False)

    print('\nRidge top 10 positive:')
    for _, r in coefs.sort_values('coef', ascending=False).head(10).iterrows():
        print(f'  {r["coef"]:+.4f}   {r["feature"]}')
    print('Ridge top 10 negative:')
    for _, r in coefs.sort_values('coef').head(10).iterrows():
        print(f'  {r["coef"]:+.4f}   {r["feature"]}')

    # ---- HistGB permutation importance ----
    print(f'\nHistGB hold-out evaluation ({time.time()-t0:.1f}s)')
    X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2, random_state=42)
    hgb = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.03, max_depth=8, l2_regularization=0.1,
        min_samples_leaf=20, random_state=42)
    hgb.fit(X_tr, y_tr)
    print(f'HistGB hold-out R2 = {hgb.score(X_te, y_te):.4f}')

    print('Computing permutation importance (n_repeats=5)...')
    perm = permutation_importance(hgb, X_te, y_te, n_repeats=5,
                                   random_state=42, n_jobs=-1, scoring='r2')
    imp_df = pd.DataFrame({
        'feature':   all_feats,
        'imp_mean':  perm.importances_mean,
        'imp_std':   perm.importances_std,
    })
    imp_df['category'] = imp_df['feature'].apply(lambda n: categorize(n, set(NUMERIC)))
    imp_df.sort_values('imp_mean', ascending=False).to_csv(
        RESULT / 'perm_importance_full.csv', index=False)
    print('\nHistGB top 15 by permutation importance:')
    for _, r in imp_df.sort_values('imp_mean', ascending=False).head(15).iterrows():
        print(f'  {r["imp_mean"]:+.5f}   {r["feature"]}')

    cat_imp = (imp_df.groupby('category')['imp_mean']
               .agg(['sum','count']).sort_values('sum', ascending=False))
    cat_imp.to_csv(RESULT / 'importance_by_category.csv')
    print('\nCategory totals:')
    print(cat_imp.to_string())
    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
