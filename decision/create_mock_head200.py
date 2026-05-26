"""
Generate a small mock head200 CSV for local testing (no Google Drive needed).

Run once:  python create_mock_head200.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DATA_DIR, MOCK_CSV

N_ROWS = 200
N_AUDIO = 80
N_GENRE = 30
RNG = np.random.default_rng(42)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = N_ROWS
    data = {
        'track_id': [f'TRMOCK{i:05d}' for i in range(n)],
        'artist_id': [f'ARMOCK{i % 40:04d}' for i in range(n)],
        'year': RNG.integers(1995, 2022, size=n),
        'duration': RNG.uniform(120_000, 280_000, size=n),
        'loudness': RNG.uniform(-18, -4, size=n),
        'tempo': RNG.uniform(70, 160, size=n),
        'artist_hotttnesss': np.clip(RNG.normal(0.4, 0.12, size=n), 0.05, 1.0),
    }
    for j in range(N_AUDIO):
        data[f'audio_feat_{j:03d}'] = RNG.normal(0, 1, size=n)
    for j in range(N_GENRE):
        data[f'genre_{j:03d}'] = (RNG.random(n) > 0.92).astype(int)

    df = pd.DataFrame(data)
    df.to_csv(MOCK_CSV, index=False)
    print(f'Wrote mock CSV ({n} rows) -> {MOCK_CSV}')


if __name__ == '__main__':
    main()
