"""
Step 19: Predict actual play count (Taste Profile) instead of song_hotttnesss.

Same song-level pipeline as step 13 but target swapped:
  step 13:  target = song_hotttnesss   (Echo Nest algorithmic score)
  step 19:  target = log1p(total_plays) (real play counts from Taste Profile)

Joins our pipeline tracks (audio + tl + genre tags + per-track year/duration)
with the per-song aggregated play counts on song_id, runs HistGB tuned with
5-fold GroupKFold by artist_id. Comparison configurations:

  audio_only_152            sanity: pure audio R^2 against play count
  strict_baseline           audio + tl + per-track yd + top-1000 genre
  full_baseline             above + 3 leak features (n_tracks_a, etc.)

Outputs:
  results/19_taste_profile_model/benchmark_taste_profile.csv
  results/19_taste_profile_model/correlation_with_song_hot.csv
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT    = Path(__file__).resolve().parent.parent
DB      = ROOT / 'data' / 'MSD_with_all_features.db'
EXTRACT = ROOT / 'data' / 'msd_summary_extract.pkl'
PLAYS   = ROOT / 'data' / 'song_play_aggregate.pkl'
HERE    = Path(__file__).resolve().parent
RESULT  = HERE / 'results' / '19_taste_profile_model'
RESULT.mkdir(exist_ok=True, parents=True)

METADATA_COLS = {'track_id','title','song_id','release','artist_id','artist_mbid',
                 'artist_name','duration','artist_familiarity','artist_hotttnesss',
                 'year','track_7digitalid','shs_perf','shs_work','term','similar'}
TL_COLS = ['loudness','tempo','key','key_confidence','mode','mode_confidence',
           'time_signature','time_signature_confidence',
           'end_of_fade_in','start_of_fade_out']

def main():
    t0 = time.time()
    plays = pd.read_pickle(PLAYS)
    extract = pd.read_pickle(EXTRACT)
    print(f'Plays per song:  {plays.shape}    ({time.time()-t0:.1f}s)')
    print(f'H5 extract:      {extract.shape}')

    # We need song_id from extract to join with plays
    extract_song = extract[['track_id','song_id','artist_id','song_hotttnesss',
                             'year','duration'] + TL_COLS].copy()

    play_song_ids = set(plays['song_id'])
    valid = extract_song[extract_song['song_id'].isin(play_song_ids)].copy()
    print(f'Tracks with play count available: {len(valid):,}')

    # Pull audio + term per song from SQLite
    cols = pd.read_sql('PRAGMA table_info(merged_partition1)',
                       sqlite3.connect(DB))['name'].tolist()
    audio_cols = [c for c in cols if c not in METADATA_COLS]
    audio_keep = [c for c in audio_cols
                  if not c.startswith('Method_of_Moments_')
                  and 'Mem20_MFCC' not in c]

    valid_track_ids = set(valid['track_id'])
    sel = ['track_id','artist_id','term'] + [f'"{c}"' for c in audio_keep]
    print('Loading audio + term from SQLite...')
    chunks = []
    with sqlite3.connect(DB) as conn:
        for ch in pd.read_sql(
            f"SELECT {', '.join(sel)} FROM merged_partition1 "
            "WHERE CAST(artist_hotttnesss AS REAL) > 0",
            conn, chunksize=100_000):
            chunks.append(ch[ch['track_id'].isin(valid_track_ids)])
    audio_df = pd.concat(chunks, ignore_index=True)
    print(f'Audio matched: {audio_df.shape}   ({time.time()-t0:.1f}s)')

    Xc = audio_df[audio_keep].values.astype(np.float64)
    keep_idx = (~np.isnan(Xc).all(axis=0)) & (np.nanstd(Xc, axis=0) > 0)
    audio_keep = [k for k, kp in zip(audio_keep, keep_idx) if kp]
    audio_df = audio_df.dropna(subset=audio_keep, how='all').reset_index(drop=True)
    print(f'Audio cols after zero-var: {len(audio_keep)}, rows: {len(audio_df)}')

    df = audio_df.merge(valid[['track_id','song_id','year','duration'] + TL_COLS],
                         on='track_id', how='inner')
    df = df.merge(plays[['song_id','total_plays','n_listeners','log_plays']],
                   on='song_id', how='inner')
    print(f'Joined audio + extract + plays: {df.shape}')

    # ---- correlation between log_plays and song_hotttnesss for sanity ----
    sh = extract_song[['song_id','song_hotttnesss']].drop_duplicates()
    cmp = df[['song_id','log_plays','total_plays']].merge(sh, on='song_id', how='left')
    cmp_valid = cmp[cmp['song_hotttnesss'] > 0]
    corrs = pd.DataFrame([
        {'pair':'log_plays vs song_hotttnesss',
         'pearson_r':  cmp_valid[['log_plays','song_hotttnesss']].corr().iloc[0,1],
         'spearman_r': cmp_valid[['log_plays','song_hotttnesss']].corr(method='spearman').iloc[0,1],
         'n': len(cmp_valid)},
        {'pair':'total_plays vs song_hotttnesss',
         'pearson_r':  cmp_valid[['total_plays','song_hotttnesss']].corr().iloc[0,1],
         'spearman_r': cmp_valid[['total_plays','song_hotttnesss']].corr(method='spearman').iloc[0,1],
         'n': len(cmp_valid)},
    ])
    corrs.to_csv(RESULT / 'correlation_with_song_hot.csv', index=False)
    print('\n=== Correlation: real plays vs Echo Nest hotttnesss ===')
    print(corrs.to_string(index=False))

    # ---- artist features ----
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
    print(f'\nGenre vocab in subset: {len(counter):,}; using top-1000')

    df.loc[df['loudness'] == 0, 'loudness'] = np.nan
    df.loc[df['tempo'] == 0, 'tempo'] = np.nan
    df.loc[df['time_signature'] == 0, 'time_signature'] = np.nan
    year_known = (df['year'] > 0).astype(np.float32).values.reshape(-1,1)
    df.loc[df['year'] == 0, 'year'] = np.nan

    Xa  = df[audio_keep].values.astype(np.float32)
    Xtl = df[TL_COLS].values.astype(np.float32)
    Xyd = np.column_stack([df['year'].values, df['duration'].values, year_known.ravel()]).astype(np.float32)
    Xex = df[['n_tracks_a','n_genres_a','year_known_ratio_a']].values.astype(np.float32)
    y   = df['log_plays'].values.astype(np.float32)
    groups = df['artist_id'].values

    print(f'\nDataset finalized: n={len(df):,}  artists={len(np.unique(groups)):,}')
    print(f'  log_plays   mean={y.mean():.3f}  std={y.std():.3f}  '
          f'min={y.min():.3f}  max={y.max():.3f}')

    cv = GroupKFold(n_splits=5)
    def cv_r2(X):
        hgb = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=8,
            l2_regularization=0.1, min_samples_leaf=20, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
        scores = cross_val_score(hgb, X, y, cv=cv, groups=groups,
                                  scoring='r2', n_jobs=-1)
        return scores.mean(), scores.std()

    rows = []
    print('\n=== HistGB R^2 on log(plays) target (5-fold GroupKFold) ===')

    X = Xa
    m, s = cv_r2(X)
    rows.append({'config':'audio_only', 'n_feat':X.shape[1], 'cv_r2':m, 'cv_std':s})
    print(f'  audio only ({X.shape[1]}):                 R^2 = {m:.4f} +/- {s:.4f}   ({time.time()-t0:.1f}s)')

    X = np.hstack([Xa, Xtl, Xyd, M])
    m, s = cv_r2(X)
    rows.append({'config':'strict_baseline', 'n_feat':X.shape[1], 'cv_r2':m, 'cv_std':s})
    print(f'  strict (audio+tl+yd+top1000): {X.shape[1]:>5d}    R^2 = {m:.4f} +/- {s:.4f}   ({time.time()-t0:.1f}s)')

    X = np.hstack([Xa, Xtl, Xyd, Xex, M])
    m, s = cv_r2(X)
    rows.append({'config':'full_baseline',   'n_feat':X.shape[1], 'cv_r2':m, 'cv_std':s})
    print(f'  full   (+ 3 leak): {X.shape[1]:>5d}              R^2 = {m:.4f} +/- {s:.4f}   ({time.time()-t0:.1f}s)')

    bench = pd.DataFrame(rows)
    bench.to_csv(RESULT / 'benchmark_taste_profile.csv', index=False)
    print(f'\n=== Saved -> {RESULT/"benchmark_taste_profile.csv"} ===')
    print(bench.to_string(index=False))

    print('\n=== Reference (target = song_hotttnesss, step 13) ===')
    print('  HistGB strict_song:  R^2 = 0.2899')
    print('  HistGB full_song:    R^2 = 0.3168')

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
