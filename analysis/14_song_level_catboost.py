"""
Step 14: Song-level CatBoost with raw `term` as text feature.

Same setup as step 13 (target = song_hotttnesss, GroupKFold by artist_id,
446,317 songs / 30,332 artists) but swap HistGB + top-1000 multi-hot
genres for CatBoost + raw `term_norm` text feature (full 7,538-tag
vocabulary, same trick that pushed artist-level R^2 from 0.428 to 0.470
in step 08).

Two configurations:

  catboost_strict_song   audio_per_track (152) + tl (10) + year + duration
                         + term as text feature (no leak)

  catboost_full_song     same + n_tracks_a, n_genres_a, year_known_ratio_a
                         (artist-level leak features replicated per song)

Outputs:
  results/14_song_level_catboost/benchmark_song_catboost.csv
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

ROOT    = Path(__file__).resolve().parent.parent
DB      = ROOT / 'data' / 'MSD_with_all_features.db'
EXTRACT = ROOT / 'data' / 'msd_summary_extract.pkl'
HERE    = Path(__file__).resolve().parent
RESULT  = HERE / 'results' / '14_song_level_catboost'
RESULT.mkdir(exist_ok=True, parents=True)

METADATA_COLS = {'track_id','title','song_id','release','artist_id','artist_mbid',
                 'artist_name','duration','artist_familiarity','artist_hotttnesss',
                 'year','track_7digitalid','shs_perf','shs_work','term','similar'}
TL_COLS = ['loudness','tempo','key','key_confidence','mode','mode_confidence',
           'time_signature','time_signature_confidence',
           'end_of_fade_in','start_of_fade_out']

def normalize_term(s):
    if pd.isna(s) or not s: return ''
    return ' '.join(t.strip().replace(' ', '_') for t in s.split(',') if t.strip())

def main():
    t0 = time.time()
    extract = pd.read_pickle(EXTRACT)
    valid = extract[extract['song_hotttnesss'] > 0].copy()
    print(f'Tracks with song_hotttnesss > 0: {len(valid):,}')

    # Pull audio + term per song (filtered to valid track ids)
    valid_ids = set(valid['track_id'])
    cols_check = pd.read_sql('PRAGMA table_info(merged_partition1)',
                             sqlite3.connect(DB))['name'].tolist()
    audio_cols = [c for c in cols_check if c not in METADATA_COLS]
    mom_prune    = [c for c in audio_cols if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_cols if 'Mem20_MFCC' in c]
    audio_keep   = [c for c in audio_cols if c not in set(mom_prune)|set(marsyas_mfcc)]

    sel = ['track_id','artist_id','term'] + [f'"{c}"' for c in audio_keep]
    print(f'Loading audio + term from SQLite...')
    chunks = []
    with sqlite3.connect(DB) as conn:
        for ch in pd.read_sql(
            f"SELECT {', '.join(sel)} FROM merged_partition1 "
            "WHERE CAST(artist_hotttnesss AS REAL) > 0",
            conn, chunksize=100_000):
            chunks.append(ch[ch['track_id'].isin(valid_ids)])
    audio_df = pd.concat(chunks, ignore_index=True)
    print(f'Loaded: {audio_df.shape}   ({time.time()-t0:.1f}s)')

    # Drop zero-var audio
    Xc = audio_df[audio_keep].values.astype(np.float64)
    keep_idx = (~np.isnan(Xc).all(axis=0)) & (np.nanstd(Xc, axis=0) > 0)
    audio_keep = [k for k, kp in zip(audio_keep, keep_idx) if kp]
    audio_df = audio_df.dropna(subset=audio_keep, how='all').reset_index(drop=True)
    print(f'Audio cols: {len(audio_keep)}   rows after NaN drop: {len(audio_df)}')

    # Merge with extract for song_hot, year, duration, tl features
    merge_cols = ['track_id','song_hotttnesss','year','duration'] + TL_COLS
    df = audio_df.merge(valid[merge_cols], on='track_id', how='inner')
    print(f'Merged: {df.shape}')

    # Per-artist aggregates for full config
    g = df.groupby('artist_id')
    artist_agg = pd.DataFrame({
        'n_tracks_a':         g.size(),
        'year_known_ratio_a': g['year'].apply(lambda s: (s > 0).mean()),
    }).reset_index()
    df = df.merge(artist_agg, on='artist_id')

    # n_genres derived from term
    df['term_norm'] = df['term'].apply(normalize_term)
    df['n_genres_a'] = df['term_norm'].apply(lambda s: len(s.split()) if s else 0)

    # Replace 0 sentinels with NaN where appropriate; CatBoost handles NaN natively
    df.loc[df['loudness'] == 0, 'loudness'] = np.nan
    df.loc[df['tempo'] == 0, 'tempo'] = np.nan
    df.loc[df['time_signature'] == 0, 'time_signature'] = np.nan
    df.loc[df['year'] == 0, 'year'] = np.nan

    y = df['song_hotttnesss'].values.astype(np.float32)
    groups = df['artist_id'].values

    cv = GroupKFold(n_splits=5)

    def cv_r2_catboost(features_no_text, text_col='term_norm'):
        # Manual CV (CatBoost's sklearn shim breaks clone() on text_features)
        X = df[features_no_text + [text_col]].copy()
        # Median-fill numeric NaN-explicit columns? CatBoost handles NaN, but only
        # for numeric features. For full transparency, leave them.
        cb_args = dict(
            iterations=500, learning_rate=0.05, depth=8, l2_leaf_reg=3,
            text_features=[text_col],
            random_seed=42, verbose=0,
            thread_count=-1, allow_writing_files=False,
        )
        scores = []
        for k, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
            t1 = time.time()
            model = CatBoostRegressor(**cb_args)
            model.fit(X.iloc[tr], y[tr])
            scr = r2_score(y[te], model.predict(X.iloc[te]))
            scores.append(scr)
            print(f'    fold {k}/5: R^2 = {scr:.4f}  ({time.time()-t1:.0f}s)')
        return float(np.mean(scores)), float(np.std(scores))

    rows = []
    print(f'\n=== Running 2 CatBoost configs ({time.time()-t0:.1f}s) ===')

    strict_features_no_text = audio_keep + TL_COLS + ['year', 'duration']
    print(f'  catboost_strict_song (audio + tl + year + dur + term_norm) '
          f'cols={len(strict_features_no_text)+1}:')
    m, s = cv_r2_catboost(strict_features_no_text)
    rows.append({'config':'strict_song', 'n_cols':len(strict_features_no_text)+1,
                 'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}   ({time.time()-t0:.1f}s)')

    full_features_no_text = strict_features_no_text + [
        'n_tracks_a','n_genres_a','year_known_ratio_a']
    print(f'\n  catboost_full_song (strict + 3 leak feats + term_norm) '
          f'cols={len(full_features_no_text)+1}:')
    m, s = cv_r2_catboost(full_features_no_text)
    rows.append({'config':'full_song', 'n_cols':len(full_features_no_text)+1,
                 'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}   ({time.time()-t0:.1f}s)')

    bench = pd.DataFrame(rows)
    bench.to_csv(RESULT / 'benchmark_song_catboost.csv', index=False)
    print(f'\n=== Saved -> {RESULT/"benchmark_song_catboost.csv"} ===')
    print(bench.to_string(index=False))

    print('\n=== Reference (song-level, target=song_hotttnesss, GroupKFold) ===')
    print('  HistGB strict_song + top-1000 (step 13):  R^2 = 0.2899')
    print('  HistGB full_song   + top-1000 (step 13):  R^2 = 0.3168')
    print('=== Reference (artist-level KFold) ===')
    print('  CatBoost full + text(7538)  (step 08):    R^2 = 0.4698')

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
