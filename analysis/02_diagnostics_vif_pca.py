"""
Step 02: Multicollinearity diagnostics + audio-only linear baselines.

Pipeline:
  1. Drop artists whose audio features are entirely NaN (~59 of ~39k)
  2. Manual feature pruning by family
       - Method_of_Moments_*  (10 cols)  is a strict subset of Area_Method_of_Moments_* (20 cols)
       - marsyas Mem20_MFCC0..12 across 4 aggregations (52 cols) duplicates the simpler
         MFCC_Overall_* (26 cols)
  3. Drop zero-variance columns
  4. Standardize, compute VIF via diag(inv(corr)), and PCA explained variance
  5. Baseline 5-fold CV R^2 on the audio-only set:
        OLS, Ridge alpha=[1,10,100], Lasso alpha=[0.001,0.01],
        OLS on top-k principal components (k = 3, 9, 18, 30, 58, 100, 152)

Outputs to results/:
  vif.csv                     VIF for every retained audio feature
  pca_explained_variance.csv  per-PC + cumulative explained variance ratio
  benchmark_audio_only.csv    R^2 for each baseline model on audio-only features
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results' / '02_diagnostics_vif_pca'
RESULT.mkdir(exist_ok=True, parents=True)

def main():
    t0 = time.time()
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    audio_feats = [c for c in audio.columns if c not in ('artist_id','hotness','n_tracks')]

    mom_prune    = [c for c in audio_feats if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_feats if 'Mem20_MFCC' in c]
    keep         = [c for c in audio_feats if c not in set(mom_prune)|set(marsyas_mfcc)]
    print(f'Manual prune: -{len(mom_prune)} (Method_of_Moments) -{len(marsyas_mfcc)} (marsyas MFCC) '
          f'= {len(keep)} remaining audio features')

    clean = audio.dropna(subset=audio_feats).reset_index(drop=True)
    print(f'Dropped {len(audio)-len(clean)} all-NaN-row artists; remaining {len(clean):,}')

    X = clean[keep].values.astype(np.float64)
    keep_idx = X.std(axis=0) > 0
    X = X[:, keep_idx]
    keep = [k for k, kp in zip(keep, keep_idx) if kp]
    print(f'After zero-variance drop: {len(keep)} features')

    Xs = StandardScaler().fit_transform(X)
    y  = clean['hotness'].values

    # ---- VIF ----
    corr = np.corrcoef(Xs, rowvar=False)
    cond = np.linalg.cond(corr)
    print(f'\ncorr matrix condition number: {cond:.2e}')
    vifs = np.diag(np.linalg.inv(corr))
    vif_df = pd.Series(vifs, index=keep).sort_values(ascending=False)
    vif_df.to_frame('vif').to_csv(RESULT / 'vif.csv')
    print(f'VIF stats:  min={vif_df.min():.2f}  median={vif_df.median():.2f}  max={vif_df.max():.2e}')
    for thr in [5, 10, 100, 1000, 1e6]:
        print(f'  VIF > {thr:>7g}:  {(vif_df > thr).sum()} / {len(vif_df)}')

    # ---- PCA ----
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame({
        'pc':              np.arange(1, len(cum)+1),
        'explained_ratio': pca.explained_variance_ratio_,
        'cum_ratio':       cum,
    })
    pca_df.to_csv(RESULT / 'pca_explained_variance.csv', index=False)
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        k = int(np.searchsorted(cum, thresh) + 1)
        print(f'PCs needed for {int(thresh*100)}% variance: {k}')

    # ---- Baseline R^2 (5-fold CV) ----
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    def cv_r2(model, Xin):
        return cross_val_score(model, Xin, y, cv=cv, scoring='r2', n_jobs=-1).mean()

    rows = [
        ('OLS',              cv_r2(LinearRegression(), Xs)),
        ('Ridge a=1',        cv_r2(Ridge(1.0),  Xs)),
        ('Ridge a=10',       cv_r2(Ridge(10.0), Xs)),
        ('Ridge a=100',      cv_r2(Ridge(100.0), Xs)),
        ('Lasso a=0.001',    cv_r2(Lasso(0.001), Xs)),
        ('Lasso a=0.01',     cv_r2(Lasso(0.01), Xs)),
    ]
    Xpc = pca.transform(Xs)
    for k in [3, 9, 18, 30, 58, 100, 152]:
        if k <= Xpc.shape[1]:
            rows.append((f'OLS top-{k} PCs', cv_r2(LinearRegression(), Xpc[:, :k])))
    bench = pd.DataFrame(rows, columns=['model', 'cv_r2_mean'])
    bench.to_csv(RESULT / 'benchmark_audio_only.csv', index=False)
    print('\n=== Audio-only benchmark (R^2, 5-fold CV) ===')
    print(bench.to_string(index=False))
    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
