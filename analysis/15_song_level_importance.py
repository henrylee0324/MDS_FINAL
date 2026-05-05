"""
Step 15: Song-level feature importance.

Step 13 produced HistGB strict_song R^2 = 0.290 / full_song = 0.317. Now
we ask "which feature blocks (and which individual features) drive that
R^2?" - i.e., does the model rely mostly on per-track audio / tl, or
on the per-artist genre tags repeated across all the artist's songs?

Design: re-run step-13 setup on a single GroupKFold split (80/20 train/
test, groups = artist_id), fit HistGB tuned, then evaluate two things on
the held-out 20%:

  (a) Block permutation importance
      Permute each feature BLOCK as a unit (preserving correlations
      within the block), record R^2 drop. Blocks: audio (152),
      tl (10), per-track year+dur+known (3), genre top-1000,
      artist_leak (3, full-only).

  (b) Single-feature permutation importance, restricted to the small
      blocks (tl, per-track meta, artist_leak), to identify which
      specific signals carry the within-artist signal.

Outputs:
  results/15_song_level_importance/block_importance.csv
  results/15_song_level_importance/single_feature_importance.csv
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

ROOT    = Path(__file__).resolve().parent.parent
DB      = ROOT / 'data' / 'MSD_with_all_features.db'
EXTRACT = ROOT / 'data' / 'msd_summary_extract.pkl'
HERE    = Path(__file__).resolve().parent
RESULT  = HERE / 'results' / '15_song_level_importance'
RESULT.mkdir(exist_ok=True, parents=True)

METADATA_COLS = {'track_id','title','song_id','release','artist_id','artist_mbid',
                 'artist_name','duration','artist_familiarity','artist_hotttnesss',
                 'year','track_7digitalid','shs_perf','shs_work','term','similar'}
TL_COLS = ['loudness','tempo','key','key_confidence','mode','mode_confidence',
           'time_signature','time_signature_confidence',
           'end_of_fade_in','start_of_fade_out']

def main():
    t0 = time.time()
    extract = pd.read_pickle(EXTRACT)
    valid = extract[extract['song_hotttnesss'] > 0].copy()
    valid_ids = set(valid['track_id'])

    cols = pd.read_sql('PRAGMA table_info(merged_partition1)',
                       sqlite3.connect(DB))['name'].tolist()
    audio_cols = [c for c in cols if c not in METADATA_COLS]
    audio_keep = [c for c in audio_cols
                  if not c.startswith('Method_of_Moments_')
                  and 'Mem20_MFCC' not in c]

    print('Loading audio + term from SQLite...')
    chunks = []
    sel = ['track_id','artist_id','term'] + [f'"{c}"' for c in audio_keep]
    with sqlite3.connect(DB) as conn:
        for ch in pd.read_sql(
            f"SELECT {', '.join(sel)} FROM merged_partition1 "
            "WHERE CAST(artist_hotttnesss AS REAL) > 0",
            conn, chunksize=100_000):
            chunks.append(ch[ch['track_id'].isin(valid_ids)])
    audio_df = pd.concat(chunks, ignore_index=True)
    print(f'Audio loaded: {audio_df.shape}   ({time.time()-t0:.1f}s)')

    Xc = audio_df[audio_keep].values.astype(np.float64)
    keep_idx = (~np.isnan(Xc).all(axis=0)) & (np.nanstd(Xc, axis=0) > 0)
    audio_keep = [k for k, kp in zip(audio_keep, keep_idx) if kp]
    audio_df = audio_df.dropna(subset=audio_keep, how='all').reset_index(drop=True)

    merge_cols = ['track_id','song_hotttnesss','year','duration'] + TL_COLS
    df = audio_df.merge(valid[merge_cols], on='track_id', how='inner')

    g = df.groupby('artist_id')
    artist_agg = pd.DataFrame({
        'n_tracks_a':         g.size(),
        'year_known_ratio_a': g['year'].apply(lambda s: (s > 0).mean()),
    }).reset_index()
    df = df.merge(artist_agg, on='artist_id')

    df['term_list'] = df['term'].fillna('').apply(
        lambda s: [t.strip() for t in s.split(',') if t.strip()])
    df['term_set']  = df['term_list'].apply(set)
    df['n_genres_a'] = df['term_list'].apply(len)

    counter = Counter()
    for lst in df['term_list']:
        counter.update(lst)
    top1000_tags = [t for t, _ in counter.most_common(1000)]
    M = np.zeros((len(df), 1000), dtype=np.float32)
    for j, tag in enumerate(top1000_tags):
        M[:, j] = df['term_set'].apply(lambda s: tag in s).values

    df.loc[df['loudness'] == 0, 'loudness'] = np.nan
    df.loc[df['tempo'] == 0, 'tempo'] = np.nan
    df.loc[df['time_signature'] == 0, 'time_signature'] = np.nan
    year_known = (df['year'] > 0).astype(np.float32).values.reshape(-1,1)
    df.loc[df['year'] == 0, 'year'] = np.nan

    Xa  = df[audio_keep].values.astype(np.float32)
    Xtl = df[TL_COLS].values.astype(np.float32)
    Xyd = np.column_stack([df['year'].values, df['duration'].values, year_known.ravel()]).astype(np.float32)
    Xex = df[['n_tracks_a','n_genres_a','year_known_ratio_a']].values.astype(np.float32)
    y   = df['song_hotttnesss'].values.astype(np.float32)
    groups = df['artist_id'].values

    # Build "full" feature stack (we'll use it for both strict and full eval)
    audio_slice  = slice(0, len(audio_keep))
    tl_slice     = slice(audio_slice.stop, audio_slice.stop + len(TL_COLS))
    yd_slice     = slice(tl_slice.stop, tl_slice.stop + 3)
    extra_slice  = slice(yd_slice.stop, yd_slice.stop + 3)
    genre_slice  = slice(extra_slice.stop, extra_slice.stop + 1000)

    X_full = np.hstack([Xa, Xtl, Xyd, Xex, M])
    print(f'Stacked X_full: {X_full.shape}   ({time.time()-t0:.1f}s)')

    feature_names = (audio_keep + TL_COLS
                     + ['year_per_track','duration_per_track','year_known_per_track']
                     + ['n_tracks_a','n_genres_a','year_known_ratio_a']
                     + [f'genre_{t}' for t in top1000_tags])

    blocks = {
        'audio (152)':              audio_slice,
        'tl (10)':                  tl_slice,
        'per_track_yd (3)':         yd_slice,
        'artist_leak (3)':          extra_slice,
        'genre_top1000 (1000)':     genre_slice,
    }

    # 5-fold GroupKFold; use first split for hold-out
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X_full, y, groups))
    X_tr, X_te = X_full[train_idx], X_full[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f'Train: {X_tr.shape}   Test: {X_te.shape}')

    print('Training HistGB tuned...')
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=8,
        l2_regularization=0.1, min_samples_leaf=50, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
    model.fit(X_tr, y_tr)
    base_r2_full = r2_score(y_te, model.predict(X_te))
    print(f'Hold-out R^2 (full set) = {base_r2_full:.4f}')

    # Strict version - same model retrained without artist_leak block
    strict_idx = np.r_[
        np.arange(audio_slice.start, audio_slice.stop),
        np.arange(tl_slice.start, tl_slice.stop),
        np.arange(yd_slice.start, yd_slice.stop),
        np.arange(genre_slice.start, genre_slice.stop)]
    print('Training HistGB tuned (strict, no artist_leak)...')
    model_strict = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=8,
        l2_regularization=0.1, min_samples_leaf=50, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
    model_strict.fit(X_tr[:, strict_idx], y_tr)
    base_r2_strict = r2_score(y_te, model_strict.predict(X_te[:, strict_idx]))
    print(f'Hold-out R^2 (strict)   = {base_r2_strict:.4f}   ({time.time()-t0:.1f}s)')

    # =================================================================
    # (a) Block permutation importance on the FULL model
    # =================================================================
    print('\n=== (a) Block permutation importance (full model, 3 repeats) ===')
    rng = np.random.RandomState(42)
    rows = []
    for name, sl in blocks.items():
        deltas = []
        for r in range(3):
            X_perm = X_te.copy()
            perm   = rng.permutation(len(X_perm))
            X_perm[:, sl] = X_perm[perm, sl]
            r2_perm = r2_score(y_te, model.predict(X_perm))
            deltas.append(base_r2_full - r2_perm)
        rows.append({'block': name, 'imp_mean': np.mean(deltas),
                     'imp_std': np.std(deltas), 'base_r2': base_r2_full})
        print(f'  {name:25s}  delta R^2 = {np.mean(deltas):+.4f} +/- {np.std(deltas):.4f}')
    pd.DataFrame(rows).to_csv(RESULT / 'block_importance.csv', index=False)

    # =================================================================
    # (b) Single-feature permutation importance for the small blocks
    #     (tl + per-track yd + artist_leak), 3 repeats
    # =================================================================
    print('\n=== (b) Single-feature permutation importance for small blocks ===')
    candidates_idx = list(range(tl_slice.start, tl_slice.stop)) \
                   + list(range(yd_slice.start, yd_slice.stop)) \
                   + list(range(extra_slice.start, extra_slice.stop))
    rows = []
    for i in candidates_idx:
        deltas = []
        for r in range(3):
            X_perm = X_te.copy()
            perm   = rng.permutation(len(X_perm))
            X_perm[:, i] = X_perm[perm, i]
            r2_perm = r2_score(y_te, model.predict(X_perm))
            deltas.append(base_r2_full - r2_perm)
        rows.append({'feature': feature_names[i], 'imp_mean': np.mean(deltas),
                     'imp_std': np.std(deltas)})
    sf = pd.DataFrame(rows).sort_values('imp_mean', ascending=False)
    sf.to_csv(RESULT / 'single_feature_importance.csv', index=False)
    print(sf.to_string(index=False))

    # =================================================================
    # (c) Audio block sub-importance: aggregate by audio sub-family
    # =================================================================
    print('\n=== (c) Audio sub-family permutation importance ===')
    def audio_subfam(name):
        if name.startswith('Area_Method'): return 'audio:Area_MoM'
        if name.startswith('LPC_'): return 'audio:LPC'
        if name.startswith('MFCC_'): return 'audio:MFCC_simple'
        if name.startswith(('Spectral_','Compactness','Root_Mean','Fraction','Zero_')):
            return 'audio:lowlevel'
        if 'Mem20' in name: return 'audio:marsyas'
        return 'audio:other'

    audio_idx = list(range(audio_slice.start, audio_slice.stop))
    audio_fams = {}
    for i in audio_idx:
        fam = audio_subfam(feature_names[i])
        audio_fams.setdefault(fam, []).append(i)

    rows = []
    for fam, idxs in audio_fams.items():
        deltas = []
        for r in range(3):
            X_perm = X_te.copy()
            perm   = rng.permutation(len(X_perm))
            X_perm[:, idxs] = X_perm[perm][:, idxs]
            r2_perm = r2_score(y_te, model.predict(X_perm))
            deltas.append(base_r2_full - r2_perm)
        rows.append({'audio_subfamily': fam, 'n_features': len(idxs),
                     'imp_mean': np.mean(deltas), 'imp_std': np.std(deltas)})
    pd.DataFrame(rows).sort_values('imp_mean', ascending=False).to_csv(
        RESULT / 'audio_subfamily_importance.csv', index=False)
    for r in sorted(rows, key=lambda x:-x['imp_mean']):
        print(f"  {r['audio_subfamily']:20s} (n={r['n_features']:>3d})  "
              f"delta R^2 = {r['imp_mean']:+.4f} +/- {r['imp_std']:.4f}")

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
