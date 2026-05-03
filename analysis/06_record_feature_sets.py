"""
Step 06: Record the exact feature set used by every modeling configuration.

This script produces three artifacts under results/ documenting "which features
went into which run":

  feature_sets_filtering.csv   pipeline of audio feature filtering, one row per
                               original feature, with columns showing which
                               filtering stage(s) kept it
  feature_sets_long.csv        long-form (script, config, feature, category)
                               table; one row per (config, feature) pair
  feature_sets_wide.csv        wide-form table where each modeling config is a
                               column and each feature is a row; cell = 1 if
                               feature is in that config
  feature_sets_summary.csv     one row per config with n_features and
                               source script
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, pandas as pd

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / 'cache'
RESULT = HERE / 'results' / '06_record_feature_sets'
RESULT.mkdir(exist_ok=True, parents=True)

NUMERIC = ['year_mean', 'duration_mean', 'year_known_ratio', 'n_tracks', 'n_genres']

def categorize(name):
    if name in NUMERIC:                     return 'numeric'
    if name.startswith('genre_'):           return 'genre'
    if name.startswith('Method_of_Moments_'): return 'audio:MoM (pruned)'
    if 'Mem20_MFCC' in name:                return 'audio:marsyas_MFCC (pruned)'
    if name.startswith('Area_Method'):      return 'audio:Area_MoM'
    if name.startswith('LPC_'):             return 'audio:LPC'
    if name.startswith('MFCC_'):            return 'audio:MFCC_simple'
    if name.startswith(('Spectral_','Compactness','Root_Mean','Fraction','Zero_')):
        return 'audio:lowlevel'
    if 'Mem20' in name:                     return 'audio:marsyas'
    return 'audio:other'

def main():
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    extra = pd.read_pickle(CACHE / 'artist_extra.pkl')

    raw_audio = [c for c in audio.columns if c not in ('artist_id','hotness','n_tracks')]
    mom_prune    = [c for c in raw_audio if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in raw_audio if 'Mem20_MFCC' in c]
    after_manual = [c for c in raw_audio if c not in set(mom_prune)|set(marsyas_mfcc)]

    clean = audio.dropna(subset=raw_audio).reset_index(drop=True)
    X_post = clean[after_manual].values.astype(np.float64)
    zero_var = [k for k, std in zip(after_manual, X_post.std(axis=0)) if std == 0]
    audio_keep = [k for k, std in zip(after_manual, X_post.std(axis=0)) if std > 0]

    print(f'Pipeline: 216 raw -> {len(after_manual)} after manual prune '
          f'-> {len(audio_keep)} after zero-var drop')
    print(f'  manual-pruned families: {len(mom_prune)} Method_of_Moments + '
          f'{len(marsyas_mfcc)} marsyas Mem20_MFCC = {len(mom_prune)+len(marsyas_mfcc)}')
    print(f'  zero-var dropped: {zero_var}')

    # ---- 1. Filtering pipeline (one row per audio feature) ----
    pipe_rows = []
    for f in raw_audio:
        manual_kept   = f not in set(mom_prune)|set(marsyas_mfcc)
        zerovar_kept  = f in audio_keep
        pipe_rows.append({
            'feature':           f,
            'category':          categorize(f),
            'in_raw_216':        1,
            'in_after_manual':   int(manual_kept),
            'in_after_zerovar':  int(zerovar_kept),
            'final_audio_152':   int(zerovar_kept),
        })
    pd.DataFrame(pipe_rows).to_csv(RESULT / 'feature_sets_filtering.csv', index=False)

    # ---- 2. All modeling configurations ----
    genre_cols = [c for c in extra.columns if c.startswith('genre_')]
    full_187   = audio_keep + NUMERIC + genre_cols
    audio_only = audio_keep
    aud_num    = audio_keep + NUMERIC
    aud_gen    = audio_keep + genre_cols
    non_audio  = NUMERIC + genre_cols
    num_only   = list(NUMERIC)
    gen_only   = genre_cols
    drop_nt    = audio_keep + ['year_mean','duration_mean','year_known_ratio','n_genres'] + genre_cols
    drop_ng    = audio_keep + ['year_mean','duration_mean','year_known_ratio','n_tracks'] + genre_cols
    strict     = audio_keep + ['year_mean','duration_mean','year_known_ratio']           + genre_cols
    very_strict= audio_keep + ['year_mean','duration_mean']                              + genre_cols

    configs = [
        ('02_diagnostics_vif_pca',    'audio_only_152',       audio_only),
        ('02_diagnostics_vif_pca',    'OLS_top3_PCs',         '<3 principal components from 152 audio>'),
        ('02_diagnostics_vif_pca',    'OLS_top9_PCs',         '<9 principal components from 152 audio>'),
        ('02_diagnostics_vif_pca',    'OLS_top18_PCs',        '<18 principal components from 152 audio>'),
        ('02_diagnostics_vif_pca',    'OLS_top30_PCs',        '<30 principal components from 152 audio>'),
        ('02_diagnostics_vif_pca',    'OLS_top58_PCs',        '<58 principal components from 152 audio>'),
        ('02_diagnostics_vif_pca',    'OLS_top100_PCs',       '<100 principal components from 152 audio>'),

        ('03_extended_baselines',     'audio_only',           audio_only),
        ('03_extended_baselines',     'audio_plus_numeric',   aud_num),
        ('03_extended_baselines',     'audio_plus_genre',     aud_gen),
        ('03_extended_baselines',     'full_187',             full_187),
        ('03_extended_baselines',     'non_audio_only',       non_audio),
        ('03_extended_baselines',     'numeric_only',         num_only),
        ('03_extended_baselines',     'genre_only',           gen_only),

        ('04_feature_importance',     'full_187',             full_187),

        ('05_leakage_check',          'full',                 full_187),
        ('05_leakage_check',          'drop_n_tracks',        drop_nt),
        ('05_leakage_check',          'drop_n_genres',        drop_ng),
        ('05_leakage_check',          'strict_drop_both',     strict),
        ('05_leakage_check',          'drop_year_known_ratio',very_strict),
        ('05_leakage_check',          'audio_plus_genre',     aud_gen),
        ('05_leakage_check',          'audio_only',           audio_only),
    ]

    # ---- 3. Long-form CSV ----
    rows = []
    for script, name, feats in configs:
        if isinstance(feats, str):
            rows.append({'script': script, 'config': name,
                         'feature': feats, 'category': 'derived'})
        else:
            for f in feats:
                rows.append({'script': script, 'config': name,
                             'feature': f, 'category': categorize(f)})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(RESULT / 'feature_sets_long.csv', index=False)

    # ---- 4. Wide-form CSV (skip PC configs since they're projections) ----
    explicit = [(s, n, f) for s, n, f in configs if not isinstance(f, str)]
    all_feats = sorted(set(f for _, _, fs in explicit for f in fs),
                      key=lambda x: (categorize(x), x))
    wide = pd.DataFrame({'feature': all_feats,
                         'category': [categorize(f) for f in all_feats]})
    for script, name, feats in explicit:
        col = f'{script.split("_")[0]}_{name}'
        wide[col] = wide['feature'].isin(set(feats)).astype(int)
    wide.to_csv(RESULT / 'feature_sets_wide.csv', index=False)

    # ---- 5. Summary ----
    summary_rows = []
    for script, name, feats in configs:
        if isinstance(feats, str):
            summary_rows.append({'script': script, 'config': name,
                                 'n_features': '(projection)', 'note': feats})
        else:
            cat_counts = pd.Series([categorize(f) for f in feats]).value_counts().to_dict()
            note = ', '.join(f'{k}:{v}' for k, v in sorted(cat_counts.items()))
            summary_rows.append({'script': script, 'config': name,
                                 'n_features': len(feats), 'note': note})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT / 'feature_sets_summary.csv', index=False)

    print('\n=== Configurations and feature counts ===')
    print(summary.to_string(index=False))
    print(f'\nSaved 4 files under {RESULT}:')
    for f in ['feature_sets_filtering.csv','feature_sets_long.csv',
              'feature_sets_wide.csv','feature_sets_summary.csv']:
        size = (RESULT / f).stat().st_size
        print(f'  {f:35s}  {size:>7d} bytes')

if __name__ == '__main__':
    main()
