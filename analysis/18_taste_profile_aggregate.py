"""
Step 18: Aggregate the MSD Taste Profile play-count triplets per song.

Input file:
  data/train_triplets.txt   (48M rows, TSV: user_id\tsong_id\tplay_count)

Per-song aggregation:
  total_plays         sum of play_count across users
  n_listeners         number of distinct users who played the song
  log_plays           log1p(total_plays)  - heavy-tailed-friendly target
  mean_per_listener   total_plays / n_listeners

Output:
  data/song_play_aggregate.pkl    one row per song_id
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIP  = ROOT / 'data' / 'train_triplets.txt.zip'
TXT  = ROOT / 'data' / 'train_triplets.txt'
OUT  = ROOT / 'data' / 'song_play_aggregate.pkl'

def main():
    t0 = time.time()
    if not TXT.exists():
        print(f'Extracting {ZIP} ...')
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(ROOT / 'data')
        print(f'Extracted in {time.time()-t0:.1f}s; size = {TXT.stat().st_size/1e9:.2f} GB')

    print('Loading triplets (chunked)...')
    chunks = []
    total_rows = 0
    for ch in pd.read_csv(TXT, sep='\t', header=None,
                          names=['user_id','song_id','play_count'],
                          dtype={'user_id':'string','song_id':'string','play_count':'int32'},
                          chunksize=2_000_000):
        chunks.append(ch.groupby('song_id', sort=False).agg(
            total_plays=('play_count','sum'),
            n_listeners=('play_count','size')
        ))
        total_rows += len(ch)
        print(f'  processed {total_rows:>11,} rows   ({time.time()-t0:.1f}s)')

    print('Combining partial aggregates...')
    combined = pd.concat(chunks).reset_index()
    final = combined.groupby('song_id', as_index=False).agg(
        total_plays=('total_plays','sum'),
        n_listeners=('n_listeners','sum'),
    )
    final['log_plays'] = np.log1p(final['total_plays'])
    final['mean_per_listener'] = final['total_plays'] / final['n_listeners']

    print(f'\nUnique songs: {len(final):,}')
    print(f'\ntotal_plays distribution:')
    print(final['total_plays'].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_string())
    print(f'\nlog_plays distribution:')
    print(final['log_plays'].describe(percentiles=[0.5, 0.9, 0.99]).to_string())

    final.to_pickle(OUT)
    print(f'\nSaved {OUT}   ({time.time()-t0:.1f}s total)')

if __name__ == '__main__':
    main()
