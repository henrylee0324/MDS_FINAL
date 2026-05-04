"""
Step 11: Diagnostic - does song_hotttnesss carry information beyond artist_hotttnesss?

If a single artist's songs all have nearly identical song_hotttnesss, then
switching from artist-level hotness to song-level hotness would change
nothing - the song-level field would just be artist-level hotness with
noise. We test this directly.

Three statistics:

  (a) Within-artist std distribution  - how much do songs by the same
                                        artist differ in hotness?

  (b) Variance decomposition (ICC)    - sigma^2(song) split into
                                        between-artist + within-artist.
                                        ICC = between / total.
                                        ICC -> 1 means artist-level captures everything;
                                        ICC -> 0 means within-artist variance dominates.

  (c) Correlation between
      mean(song_hot per artist) and artist_hotttnesss

      If they are essentially the same number, artist_hotttnesss is just
      an aggregate of song_hotttnesss and switching targets adds info only
      via the within-artist component.

Outputs:
  results/11_song_vs_artist_hotness/diagnostics.csv
  results/11_song_vs_artist_hotness/per_artist_stats.csv
  results/11_song_vs_artist_hotness/extreme_cases.csv  - top hidden-hit + dud-track artists
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np, pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
HERE   = Path(__file__).resolve().parent
RESULT = HERE / 'results' / '11_song_vs_artist_hotness'
RESULT.mkdir(exist_ok=True, parents=True)
EXTRACT = ROOT / 'data' / 'msd_summary_extract.pkl'

def main():
    t0 = time.time()
    df = pd.read_pickle(EXTRACT)
    print(f'Loaded extract: {df.shape}   ({time.time()-t0:.1f}s)')

    # Keep songs with both song_hotttnesss > 0 AND artist_hotttnesss > 0
    valid = df[(df['song_hotttnesss'] > 0) & (df['artist_hotttnesss'] > 0)].copy()
    print(f'Songs with both hotness > 0: {len(valid):,}')

    # Per-artist aggregation
    g = valid.groupby('artist_id')
    per_artist = pd.DataFrame({
        'n_songs':         g.size(),
        'song_hot_mean':   g['song_hotttnesss'].mean(),
        'song_hot_median': g['song_hotttnesss'].median(),
        'song_hot_std':    g['song_hotttnesss'].std(),
        'song_hot_min':    g['song_hotttnesss'].min(),
        'song_hot_max':    g['song_hotttnesss'].max(),
        'song_hot_range':  g['song_hotttnesss'].max() - g['song_hotttnesss'].min(),
        'artist_hot':      g['artist_hotttnesss'].first(),
    }).reset_index()
    print(f'Per-artist stats: {per_artist.shape}')
    per_artist.to_csv(RESULT / 'per_artist_stats.csv', index=False)

    # Restrict to artists with >= 5 valid songs for meaningful within-artist stats
    multi = per_artist[per_artist['n_songs'] >= 5].copy()
    print(f'Artists with >=5 valid songs: {len(multi):,}')

    # ======== (a) Within-artist std distribution ========
    print('\n=== (a) Within-artist std distribution ===')
    print(multi['song_hot_std'].describe().to_string())
    print(f'  artists with std < 0.05 (essentially constant): '
          f'{(multi["song_hot_std"] < 0.05).sum():,} ({(multi["song_hot_std"] < 0.05).mean()*100:.1f}%)')
    print(f'  artists with std > 0.15 (genuinely varies):     '
          f'{(multi["song_hot_std"] > 0.15).sum():,} ({(multi["song_hot_std"] > 0.15).mean()*100:.1f}%)')

    # ======== (b) Variance decomposition (ICC) ========
    print('\n=== (b) Variance decomposition (ICC) ===')
    artists_used = set(multi['artist_id'])
    sub = valid[valid['artist_id'].isin(artists_used)].copy()
    overall_mean = sub['song_hotttnesss'].mean()
    artist_means = sub.groupby('artist_id')['song_hotttnesss'].mean()
    n_per = sub.groupby('artist_id').size()

    # Between-artist variance (weighted by n_per to be fair)
    between_var = ((artist_means - overall_mean)**2 * n_per).sum() / n_per.sum()
    # Within-artist variance (pooled)
    sub = sub.merge(artist_means.rename('artist_mean_hot'), on='artist_id')
    within_var = ((sub['song_hotttnesss'] - sub['artist_mean_hot'])**2).mean()
    total_var  = ((sub['song_hotttnesss'] - overall_mean)**2).mean()
    icc = between_var / total_var

    print(f'  overall mean hotness: {overall_mean:.4f}')
    print(f'  total variance:    {total_var:.5f}')
    print(f'  between-artist:    {between_var:.5f}  ({between_var/total_var*100:.1f}%)')
    print(f'  within-artist:     {within_var:.5f}  ({within_var/total_var*100:.1f}%)')
    print(f'  ICC = {icc:.4f}')
    if icc > 0.7:
        print('  -> artist_hotttnesss already captures most of the song-level variability')
    elif icc > 0.4:
        print('  -> meaningful within-artist signal exists (worth modeling at song level)')
    else:
        print('  -> within-artist variance dominates (song level matters a lot)')

    # ======== (c) Correlation: mean(song_hot per artist) vs artist_hotttnesss ========
    print('\n=== (c) Per-artist mean(song_hot) vs artist_hotttnesss ===')
    corr_pearson  = multi[['song_hot_mean','artist_hot']].corr().iloc[0,1]
    corr_spearman = multi[['song_hot_mean','artist_hot']].corr(method='spearman').iloc[0,1]
    diff = multi['song_hot_mean'] - multi['artist_hot']
    print(f'  Pearson r:   {corr_pearson:.4f}')
    print(f'  Spearman r:  {corr_spearman:.4f}')
    print(f'  diff (song_mean - artist_hot):  mean={diff.mean():+.4f}  std={diff.std():.4f}')
    print(f'  |diff| > 0.05:  {(diff.abs() > 0.05).sum():,} of {len(multi):,} '
          f'({(diff.abs() > 0.05).mean()*100:.1f}%)')

    # ======== (d) Extreme cases: hidden-hit and dud-track artists ========
    print('\n=== (d) Extreme cases ===')
    multi['hidden_hit_score'] = multi['song_hot_max'] - multi['artist_hot']
    multi['dud_track_score']  = multi['artist_hot'] - multi['song_hot_min']

    hidden_hits = multi.nlargest(15, 'hidden_hit_score')[
        ['artist_id','n_songs','artist_hot','song_hot_max','song_hot_min','hidden_hit_score']]
    duds = multi.nlargest(15, 'dud_track_score')[
        ['artist_id','n_songs','artist_hot','song_hot_max','song_hot_min','dud_track_score']]

    print('\n  Top 10 "hidden hit" artists (highest song hotness >> artist hotness):')
    print(hidden_hits.head(10).to_string(index=False))
    print('\n  Top 10 "dud track" artists (artist hotness >> lowest song):')
    print(duds.head(10).to_string(index=False))

    pd.concat([hidden_hits.assign(kind='hidden_hit'),
               duds.assign(kind='dud_track')]).to_csv(
        RESULT / 'extreme_cases.csv', index=False)

    # Summary CSV
    diag = pd.DataFrame([
        {'metric':'n_songs (both hotness valid)',           'value': len(valid)},
        {'metric':'n_artists (>=1 valid song)',             'value': len(per_artist)},
        {'metric':'n_artists (>=5 valid songs, used here)', 'value': len(multi)},
        {'metric':'within-std median',                       'value': multi['song_hot_std'].median()},
        {'metric':'within-std p75',                          'value': multi['song_hot_std'].quantile(0.75)},
        {'metric':'between-artist variance share',           'value': between_var/total_var},
        {'metric':'within-artist variance share',            'value': within_var/total_var},
        {'metric':'ICC',                                     'value': icc},
        {'metric':'pearson(song_mean, artist_hot)',          'value': corr_pearson},
        {'metric':'spearman(song_mean, artist_hot)',         'value': corr_spearman},
    ])
    diag.to_csv(RESULT / 'diagnostics.csv', index=False)
    print(f'\nSaved diagnostics + per-artist + extreme-cases CSVs to {RESULT}/')
    print(f'Done in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
