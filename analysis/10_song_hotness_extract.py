"""
Step 10: Extract song_hotttnesss + track-level Echo Nest analysis features
from the official MSD summary file.

The pre-processed SQLite DB used by steps 01-09 had `song_hotttnesss`
removed. The ~316 MB `msd_summary_file.h5` from millionsongdataset.com
contains it for all 1,000,000 tracks (along with track-level audio
analysis features that are NOT in our SQLite).

Download URL:
  http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/msd_summary_file.h5
  (redirects to millionsongdataset.com; ~316 MB; expects pytables)

Outputs:
  data/msd_summary_extract.pkl     1M rows x 20 cols, 321 MB in memory
                                   contains track_id, song_id, artist_id,
                                   song_hotttnesss, artist_hotttnesss,
                                   year, plus Echo Nest analysis features
                                   (loudness, tempo, key, mode, time_signature,
                                    duration, fade times)
  results/10_song_hotness_extract/
    coverage_report.csv            row counts and NaN/zero stats per column
    join_with_artist_pipeline.csv  overlap stats vs the pipeline used by
                                   steps 01-09 (artist-level filtered set)

Usage notes:
  danceability and energy are present in the H5 but ALL ZEROS
  (well-known: Echo Nest never populated these for MSD). Skipped here.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np, pandas as pd
import tables, sqlite3

ROOT   = Path(__file__).resolve().parent.parent
H5     = ROOT / 'data' / 'msd_summary_file.h5'
DB     = ROOT / 'data' / 'MSD_with_all_features.db'
HERE   = Path(__file__).resolve().parent
RESULT = HERE / 'results' / '10_song_hotness_extract'
RESULT.mkdir(exist_ok=True, parents=True)

ANALYSIS_COLS = ['loudness', 'tempo', 'key', 'key_confidence',
                 'mode', 'mode_confidence',
                 'time_signature', 'time_signature_confidence',
                 'duration', 'end_of_fade_in', 'start_of_fade_out']

def main():
    t0 = time.time()
    if not H5.exists():
        raise SystemExit(f'Missing {H5}. Download with:\n'
            '  curl -L -o data/msd_summary_file.h5 '
            '"http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/msd_summary_file.h5"')

    print(f'Reading {H5} ({H5.stat().st_size/1e6:.1f} MB)...')
    with tables.open_file(str(H5), 'r') as h5:
        a  = h5.root.analysis.songs
        m  = h5.root.metadata.songs
        mb = h5.root.musicbrainz.songs
        df = pd.DataFrame({
            'track_id':           [b.decode() for b in a.col('track_id')],
            'song_id':            [b.decode() for b in m.col('song_id')],
            'artist_id':          [b.decode() for b in m.col('artist_id')],
            'song_hotttnesss':    m.col('song_hotttnesss'),
            'artist_hotttnesss':  m.col('artist_hotttnesss'),
            'artist_familiarity': m.col('artist_familiarity'),
            'year':               mb.col('year'),
            **{c: a.col(c) for c in ANALYSIS_COLS},
        })
    print(f'Extracted: {df.shape}   ({time.time()-t0:.1f}s)   '
          f'mem={df.memory_usage(deep=True).sum()/1e6:.1f} MB')

    # Coverage report
    cov_rows = []
    for col in ['song_hotttnesss', 'artist_hotttnesss', 'artist_familiarity', 'year'] + ANALYSIS_COLS:
        v = df[col]
        cov_rows.append({
            'column': col,
            'n_total': len(v),
            'n_nan':   int(v.isna().sum()),
            'n_zero':  int((v == 0).sum()),
            'n_pos':   int((v > 0).sum()),
            'mean':    float(v[v > 0].mean()) if (v > 0).any() else np.nan,
            'median':  float(v[v > 0].median()) if (v > 0).any() else np.nan,
        })
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(RESULT / 'coverage_report.csv', index=False)
    print('\n=== Coverage report (n_pos = entries with value > 0) ===')
    print(cov.to_string(index=False))

    # Join check vs the artist-level pipeline used by steps 01-09
    with sqlite3.connect(DB) as conn:
        pipeline_artists = pd.read_sql("""
            SELECT DISTINCT artist_id
            FROM merged_partition1
            WHERE CAST(artist_hotttnesss AS REAL) > 0
        """, conn)
        pipeline_tracks = pd.read_sql("""
            SELECT DISTINCT track_id
            FROM merged_partition1
            WHERE CAST(artist_hotttnesss AS REAL) > 0
        """, conn)
    print(f'\nPipeline artists (hotness>0): {len(pipeline_artists):,}')
    print(f'Pipeline tracks  (hotness>0): {len(pipeline_tracks):,}')

    # Tracks with valid song_hotttnesss
    valid_song = df[df['song_hotttnesss'] > 0].copy()
    print(f'H5 tracks with song_hotttnesss > 0: {len(valid_song):,}')

    # Intersect
    pipeline_track_ids = set(pipeline_tracks['track_id'])
    overlap_tracks = valid_song[valid_song['track_id'].isin(pipeline_track_ids)]
    print(f'Overlap (pipeline track AND song_hotttnesss>0): {len(overlap_tracks):,}')

    # Per-artist: how many artists have at least 1 song with valid song_hotttnesss?
    pipeline_artist_ids = set(pipeline_artists['artist_id'])
    artists_with_song_hot = (valid_song[valid_song['artist_id'].isin(pipeline_artist_ids)]
                             .groupby('artist_id').size())
    print(f'\nPipeline artists with >=1 song having song_hotttnesss: {len(artists_with_song_hot):,}')
    print(f'                                       >=5 such songs: {(artists_with_song_hot >= 5).sum():,}')
    print(f'                                       >=10 such songs: {(artists_with_song_hot >= 10).sum():,}')

    join_rows = [
        {'metric': 'pipeline artists (artist_hotttnesss > 0)',
         'value':  len(pipeline_artists)},
        {'metric': 'pipeline tracks  (artist_hotttnesss > 0)',
         'value':  len(pipeline_tracks)},
        {'metric': 'H5 tracks with song_hotttnesss > 0',
         'value':  len(valid_song)},
        {'metric': 'tracks in BOTH (overlap)',
         'value':  len(overlap_tracks)},
        {'metric': 'artists with >= 1 song hotness',
         'value':  len(artists_with_song_hot)},
        {'metric': 'artists with >= 5 song hotness',
         'value':  int((artists_with_song_hot >= 5).sum())},
        {'metric': 'artists with >= 10 song hotness',
         'value':  int((artists_with_song_hot >= 10).sum())},
    ]
    pd.DataFrame(join_rows).to_csv(RESULT / 'join_with_artist_pipeline.csv', index=False)

    out = ROOT / 'data' / 'msd_summary_extract.pkl'
    df.to_pickle(out)
    print(f'\nSaved {out} ({out.stat().st_size/1e6:.1f} MB on disk)')
    print(f'Done in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
