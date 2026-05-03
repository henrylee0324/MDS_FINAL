"""
Step 01: Aggregate per-artist features.

Reads `data/MSD_with_all_features.db` and produces two cached DataFrames:

  cache/artist_audio_agg.pkl   audio features averaged per artist
                               (216 cols + artist_id + hotness + n_tracks)
  cache/artist_extra.pkl       year_mean, duration_mean, year_known_ratio,
                               n_genres, plus top-30 genre multi-hot indicators

Filtering: only tracks with `artist_hotttnesss > 0` are aggregated
(removes -1 and 0 sentinels for missing scores).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT  = Path(__file__).resolve().parent.parent
DB    = ROOT / 'data' / 'MSD_with_all_features.db'
CACHE = Path(__file__).resolve().parent / 'cache'
CACHE.mkdir(exist_ok=True)

METADATA = {'track_id','title','song_id','release','artist_id','artist_mbid',
            'artist_name','duration','artist_familiarity','artist_hotttnesss',
            'year','track_7digitalid','shs_perf','shs_work','term','similar'}

def aggregate_audio(conn):
    cols = pd.read_sql('PRAGMA table_info(merged_partition1)', conn)['name'].tolist()
    feats = [c for c in cols if c not in METADATA]
    sel_feats = ', '.join(f'AVG("{c}") AS "{c}"' for c in feats)
    q = f"""
        SELECT artist_id,
               AVG(CAST(artist_hotttnesss AS REAL)) AS hotness,
               COUNT(*) AS n_tracks,
               {sel_feats}
        FROM merged_partition1
        WHERE CAST(artist_hotttnesss AS REAL) > 0
        GROUP BY artist_id
    """
    return pd.read_sql(q, conn)

def aggregate_extra(conn, top_k_genres=30):
    extra = pd.read_sql("""
        SELECT artist_id,
               AVG(CASE WHEN CAST(year AS INTEGER) > 0 THEN CAST(year AS INTEGER) END) AS year_mean,
               AVG(CAST(duration AS REAL)) AS duration_mean,
               SUM(CASE WHEN CAST(year AS INTEGER) > 0 THEN 1 ELSE 0 END) * 1.0
                 / COUNT(*) AS year_known_ratio,
               MAX(term) AS term
        FROM merged_partition1
        WHERE CAST(artist_hotttnesss AS REAL) > 0
        GROUP BY artist_id
    """, conn)

    extra['term_list'] = extra['term'].fillna('').apply(
        lambda s: [t.strip() for t in s.split(',') if t.strip()])
    counter = Counter()
    for lst in extra['term_list']:
        counter.update(lst)
    top = [t for t, _ in counter.most_common(top_k_genres)]
    for g in top:
        extra[f'genre_{g}'] = extra['term_list'].apply(lambda lst: int(g in lst))
    extra['n_genres'] = extra['term_list'].apply(len)
    return extra.drop(columns=['term', 'term_list']), top

def main():
    t0 = time.time()
    conn = sqlite3.connect(DB)

    print('Aggregating audio features per artist (SQL AVG)...')
    audio = aggregate_audio(conn)
    audio.to_pickle(CACHE / 'artist_audio_agg.pkl')
    print(f'  shape={audio.shape}  saved -> {CACHE/"artist_audio_agg.pkl"}')
    print(f'  hotness mean={audio["hotness"].mean():.3f}  std={audio["hotness"].std():.3f}')
    print(f'  tracks/artist median={int(audio["n_tracks"].median())}  max={int(audio["n_tracks"].max())}')

    print('\nAggregating extra (year/duration) + top-30 genre tags...')
    extra, top_genres = aggregate_extra(conn, top_k_genres=30)
    extra.to_pickle(CACHE / 'artist_extra.pkl')
    print(f'  shape={extra.shape}  saved -> {CACHE/"artist_extra.pkl"}')
    print(f'  artists with year_mean missing: {extra["year_mean"].isna().sum()}')
    print(f'  top 10 genres: {top_genres[:10]}')

    conn.close()
    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
