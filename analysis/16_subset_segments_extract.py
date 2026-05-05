"""
Step 16: Extract segment-level features from the MSD 10K subset.

The subset's per-song HDF5 files contain Echo Nest analysis arrays
(segments_timbre, segments_pitches, segments_loudness_*, bars_*,
beats_*, sections_*, tatums_*) that are NOT in the summary file we used
in step 10. This script walks all 10K HDF5 files, derives a fixed-size
feature vector per song, and saves a single DataFrame.

Derived features per song (101 total):

  Block 1 - Timbre summary (24)
    For each of 12 timbre dims: mean, std

  Block 2 - Timbre temporal evolution (36)
    Mean per timbre dim in first / middle / last third of segments
    (3 * 12 = 36)

  Block 3 - Pitches/chroma summary (24)
    For each of 12 chroma dims: mean, std
    (Mean chroma vector ~ overall pitch class distribution / key signature)

  Block 4 - Loudness dynamics (7)
    segments_loudness_max stats: mean, std, min, max, range, plus mean
    of first / last third

  Block 5 - Rhythm regularity (7)
    Inter-beat interval mean & CV
    Inter-bar interval mean & CV
    Inter-tatum interval mean
    n_segments
    n_beats_per_bar

  Block 6 - Structure (3)
    n_sections, mean section length, std section length

Output:
  data/subset_segment_features.pkl   ~10k rows x 101 derived feats + track_id
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np, pandas as pd
import tables, glob

ROOT   = Path(__file__).resolve().parent.parent
SUBSET = ROOT / 'data' / 'MillionSongSubset'  # extract destination guess
OUT    = ROOT / 'data' / 'subset_segment_features.pkl'

def featurize(h5_path):
    with tables.open_file(str(h5_path), 'r') as h5:
        a = h5.root.analysis
        track_id = a.songs.col('track_id')[0].decode()
        timbre = a.segments_timbre.read()       # (n_seg, 12)
        pitches= a.segments_pitches.read()      # (n_seg, 12)
        sl_max = a.segments_loudness_max.read() # (n_seg,)
        beats  = a.beats_start.read()           # (n_beats,)
        bars   = a.bars_start.read()            # (n_bars,)
        tatums = a.tatums_start.read()          # (n_tatums,)
        sections = a.sections_start.read()      # (n_sec,)

    n_seg = max(len(timbre), 1)
    f = {'track_id': track_id, 'n_segments': n_seg}

    # Block 1: timbre mean/std per dim
    if n_seg > 0:
        for d in range(12):
            f[f'tim_mean_{d}'] = float(timbre[:, d].mean())
            f[f'tim_std_{d}']  = float(timbre[:, d].std())
    else:
        for d in range(12):
            f[f'tim_mean_{d}'] = 0.0
            f[f'tim_std_{d}']  = 0.0

    # Block 2: timbre temporal evolution (first / middle / last third)
    third = max(n_seg // 3, 1)
    splits = [(0, third), (third, 2*third), (2*third, n_seg)]
    for i, (s, e) in enumerate(splits):
        seg = timbre[s:e] if n_seg > 0 else np.zeros((1, 12))
        seg = seg if len(seg) > 0 else np.zeros((1, 12))
        for d in range(12):
            f[f'tim_third{i}_mean_{d}'] = float(seg[:, d].mean())

    # Block 3: chroma mean/std per dim
    if n_seg > 0:
        for d in range(12):
            f[f'chr_mean_{d}'] = float(pitches[:, d].mean())
            f[f'chr_std_{d}']  = float(pitches[:, d].std())
    else:
        for d in range(12):
            f[f'chr_mean_{d}'] = 0.0
            f[f'chr_std_{d}']  = 0.0

    # Block 4: loudness dynamics
    if len(sl_max) > 0:
        f['loud_mean']  = float(sl_max.mean())
        f['loud_std']   = float(sl_max.std())
        f['loud_min']   = float(sl_max.min())
        f['loud_max']   = float(sl_max.max())
        f['loud_range'] = float(sl_max.max() - sl_max.min())
        third_sl = max(len(sl_max) // 3, 1)
        f['loud_first_third_mean'] = float(sl_max[:third_sl].mean())
        f['loud_last_third_mean']  = float(sl_max[-third_sl:].mean())
    else:
        for k in ['loud_mean','loud_std','loud_min','loud_max','loud_range',
                  'loud_first_third_mean','loud_last_third_mean']:
            f[k] = np.nan

    # Block 5: rhythm regularity
    def iois(arr):
        return np.diff(arr) if len(arr) > 1 else np.array([np.nan])
    bi = iois(beats)
    f['beat_ioi_mean'] = float(np.nanmean(bi))
    f['beat_ioi_cv']   = float(np.nanstd(bi) / max(abs(np.nanmean(bi)), 1e-9))
    bri = iois(bars)
    f['bar_ioi_mean']  = float(np.nanmean(bri))
    f['bar_ioi_cv']    = float(np.nanstd(bri) / max(abs(np.nanmean(bri)), 1e-9))
    f['tatum_ioi_mean']= float(np.nanmean(iois(tatums)))
    f['n_beats_per_bar'] = (len(beats) / len(bars)) if len(bars) > 0 else np.nan
    # n_segments already added

    # Block 6: structure
    f['n_sections'] = len(sections)
    if len(sections) > 1:
        sec_lens = np.diff(sections)
        f['sec_len_mean'] = float(sec_lens.mean())
        f['sec_len_std']  = float(sec_lens.std())
    else:
        f['sec_len_mean'] = np.nan
        f['sec_len_std']  = np.nan

    return f

def main():
    t0 = time.time()
    paths = list(SUBSET.rglob('*.h5'))
    print(f'Found {len(paths)} h5 files under {SUBSET}')
    if not paths:
        print('No files found. Make sure the subset has been extracted.')
        return

    rows, errors = [], 0
    for i, p in enumerate(paths):
        try:
            rows.append(featurize(p))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'  ERR {p.name}: {e}')
        if (i+1) % 1000 == 0:
            print(f'  processed {i+1:>5d} / {len(paths)}   ({time.time()-t0:.1f}s)')

    df = pd.DataFrame(rows)
    df.to_pickle(OUT)
    print(f'\nSaved {OUT}')
    print(f'  rows={len(df):,}   cols={df.shape[1]}   errors={errors}')
    print(f'  done in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
