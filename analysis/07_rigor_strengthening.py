"""
Step 07: Rigor strengthening for the feature-selection methodology.

Three additions to back the manual / structural pruning done in steps 02-05
with algorithmic / statistical evidence:

A. Iterative VIF pruning on the 152-audio set
   Drop the highest-VIF feature one at a time until all VIF < 10.
   Record surviving features and the resulting Ridge / OLS R^2.

B. Lasso active set sweep
   Fit Lasso at several alphas on standardized 152-audio; record which
   features get non-zero coefficients at each alpha and the corresponding
   5-fold CV R^2.

C. Stratified-by-hotness 5-fold CV
   Re-evaluate the four headline configurations (Ridge / HistGB on full
   187 and strict 185) using StratifiedKFold over hotness quintiles
   instead of random KFold.

Outputs to results/:
  vif_iterative_trace.csv         per-iteration record (n_left, max_vif, dropped)
  vif_iterative_survivors.csv     features that survived VIF<10 cutoff
  lasso_active_sets.csv           which features active at each alpha
  stratified_cv_benchmark.csv     R^2 mean/std under hotness-stratified CV
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Lasso, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results' / '07_rigor_strengthening'
RESULT.mkdir(exist_ok=True, parents=True)

NUMERIC = ['year_mean','duration_mean','year_known_ratio','n_tracks','n_genres']

def main():
    t0 = time.time()
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    extra = pd.read_pickle(CACHE / 'artist_extra.pkl')

    audio_feats = [c for c in audio.columns if c not in ('artist_id','hotness','n_tracks')]
    mom_prune    = [c for c in audio_feats if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_feats if 'Mem20_MFCC' in c]
    audio_keep   = [c for c in audio_feats if c not in set(mom_prune)|set(marsyas_mfcc)]

    clean = audio.dropna(subset=audio_feats).reset_index(drop=True)
    Xa = clean[audio_keep].values.astype(np.float64)
    audio_keep = [k for k, kp in zip(audio_keep, Xa.std(axis=0)>0) if kp]
    df = clean.merge(extra, on='artist_id', how='left')
    y = df['hotness'].values

    Xs_audio = StandardScaler().fit_transform(df[audio_keep].values.astype(np.float64))
    print(f'Starting set: {Xs_audio.shape[1]} audio features, n={len(y)}')

    # =================================================================
    # A. Iterative VIF pruning
    # =================================================================
    print('\n=== A. Iterative VIF pruning (drop max-VIF until all < 10) ===')
    corr_full = np.corrcoef(Xs_audio, rowvar=False)
    keep_mask = np.ones(corr_full.shape[0], dtype=bool)
    trace = []
    iter_no = 0
    while True:
        iter_no += 1
        sub = corr_full[keep_mask][:, keep_mask]
        try:
            inv = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(sub)
        vifs = np.diag(inv)
        max_v, max_idx_local = vifs.max(), vifs.argmax()
        sub_indices = np.where(keep_mask)[0]
        dropped_idx = sub_indices[max_idx_local]
        dropped_name = audio_keep[dropped_idx]
        trace.append({
            'iter':      iter_no,
            'n_remaining': int(keep_mask.sum()),
            'max_vif':   float(max_v),
            'dropped':   dropped_name if max_v >= 10 else None,
        })
        if max_v < 10 or keep_mask.sum() <= 2:
            break
        keep_mask[dropped_idx] = False
    survivors = [audio_keep[i] for i in np.where(keep_mask)[0]]
    print(f'  iterations: {iter_no}    survivors: {len(survivors)}')
    print(f'  final max VIF: {trace[-1]["max_vif"]:.2f}')

    pd.DataFrame(trace).to_csv(RESULT / 'vif_iterative_trace.csv', index=False)
    pd.Series(survivors, name='feature').to_csv(
        RESULT / 'vif_iterative_survivors.csv', index=False)

    # R^2 on the VIF-pruned set
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    Xs_vif = Xs_audio[:, keep_mask]
    r2_ols   = cross_val_score(LinearRegression(), Xs_vif, y, cv=cv, scoring='r2', n_jobs=-1).mean()
    r2_ridge = cross_val_score(RidgeCV(alphas=[0.1,1,10,100,1000]),
                                Xs_vif, y, cv=cv, scoring='r2', n_jobs=-1).mean()
    print(f'  R^2 on the {len(survivors)} VIF-survivors: OLS={r2_ols:.4f}  Ridge={r2_ridge:.4f}')
    print(f'  (recall: full 152 audio Ridge alpha=100 = 0.084 from step 02)')

    # =================================================================
    # B. Lasso active set sweep
    # =================================================================
    print(f'\n=== B. Lasso active set sweep ({time.time()-t0:.1f}s) ===')
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    rows = []
    for a in alphas:
        las = Lasso(alpha=a, max_iter=20000)
        las.fit(Xs_audio, y)
        active = [audio_keep[i] for i, c in enumerate(las.coef_) if c != 0]
        r2 = cross_val_score(Lasso(alpha=a, max_iter=20000),
                              Xs_audio, y, cv=cv, scoring='r2', n_jobs=-1).mean()
        rows.append({'alpha': a, 'n_active': len(active), 'cv_r2': r2,
                     'active_features': '|'.join(active)})
        print(f'  alpha={a:>7g}: {len(active):>4d} active, R^2 = {r2:.4f}')
    pd.DataFrame(rows).to_csv(RESULT / 'lasso_active_sets.csv', index=False)

    # =================================================================
    # C. Stratified-by-hotness CV
    # =================================================================
    print(f'\n=== C. Stratified-by-hotness 5-fold CV ({time.time()-t0:.1f}s) ===')
    hot_bin = pd.qcut(y, q=5, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    full_feats   = audio_keep + NUMERIC + genre_cols
    strict_feats = audio_keep + ['year_mean','duration_mean','year_known_ratio'] + genre_cols

    def build(feat_list):
        return df[feat_list].values.astype(np.float64)

    def evaluate(X, model_kind, splits, name, n_feat):
        if model_kind == 'ridge':
            n_to_scale = sum(1 for f in (full_feats if n_feat == len(full_feats) else strict_feats)
                             if not f.startswith('genre_'))
            ct = ColumnTransformer([
                ('aud_num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                      ('sc',  StandardScaler())]),
                 list(range(n_to_scale))),
                ('gen',     'passthrough', list(range(n_to_scale, X.shape[1]))),
            ])
            model = Pipeline([('ct', ct), ('reg', RidgeCV(alphas=[0.1,1,10,100,1000]))])
        else:
            model = HistGradientBoostingRegressor(
                max_iter=600, learning_rate=0.03, max_depth=8,
                l2_regularization=0.1, min_samples_leaf=20, random_state=42)
        scores = cross_val_score(model, X, y, cv=splits, scoring='r2', n_jobs=-1)
        return scores.mean(), scores.std()

    splits = list(skf.split(np.zeros((len(y),1)), hot_bin))
    rng_splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros((len(y),1))))

    rows = []
    for cfg_name, feat_list in [('full_187', full_feats), ('strict_185', strict_feats)]:
        X = build(feat_list)
        for model_name in ['ridge', 'hgb']:
            mean_strat, std_strat = evaluate(X, model_name, splits,    cfg_name, len(feat_list))
            mean_rand,  std_rand  = evaluate(X, model_name, rng_splits, cfg_name, len(feat_list))
            rows.append({
                'config':        cfg_name,
                'model':         model_name,
                'random_kfold':  mean_rand,  'random_std':  std_rand,
                'stratified_cv': mean_strat, 'stratified_std': std_strat,
                'delta':         mean_strat - mean_rand,
            })
            print(f'  {cfg_name:11s} {model_name:5s}  random={mean_rand:.4f}+/-{std_rand:.4f}  '
                  f'strat={mean_strat:.4f}+/-{std_strat:.4f}  delta={mean_strat-mean_rand:+.4f}')
    pd.DataFrame(rows).to_csv(RESULT / 'stratified_cv_benchmark.csv', index=False)

    print(f'\nDone in {time.time()-t0:.1f}s. Saved 4 files under {RESULT}/')

if __name__ == '__main__':
    main()
