"""Column names and business / perturbation config."""

TARGET_CANDIDATES = [
    'song_hotttnesss', 'song_hotttness', 'artist_hotttnesss', 'artist_hotttness', 'hotness',
]

ID_COLS = {
    'track_id', 'song_id', 'title', 'release', 'artist_id', 'artist_mbid',
    'artist_name', 'similar', 'analyzer_version', 'new_song_id', 'profile_name',
    'base_row_index', 'source_track_id_masked',
}

# 聲學約束可用欄位（PDF 式 3–4：檔期平均門檻）
ACOUSTIC_METRICS = ['loudness', 'tempo', 'danceability', 'energy']
ACOUSTIC_FOR_CONSTRAINTS = ACOUSTIC_METRICS

BASE_ROW_INDEX = 0
N_SYNTHETIC_SONGS = 20

# FLOP–HOP–TOP 分級（僅用於傳統對照標籤，成本以 COST_BY_PROFILE 為準）
COST_TIERS = [
    {'name': 'TOP', 'y_min': 0.58, 'cost': 55.0},
    {'name': 'HOP', 'y_min': 0.40, 'cost': 32.0},
    {'name': 'FLOP', 'y_min': 0.0, 'cost': 14.0},
]

# 每首模擬歌的「行銷+製作」成本（萬元假設單位，刻意拉開差異）
COST_BY_PROFILE = {
    '夏日高能量': 58, '抒情慢板': 28, '復古翻新': 35, '最新潮流': 52, '短歌衝刺': 48,
    '長曲敘事': 38, '舞曲場': 56, '民謠感': 22, '主流中庸': 42, '實驗邊緣': 45,
    '電音派對': 60, '原聲不插電': 20, '懷舊金曲': 30, '街頭嘻哈': 50, '電影配樂感': 36,
    '兒童節慶': 40, '金屬衝擊': 54, '爵士深夜': 26, '民謠清新': 24, '合成器流行': 46,
}

# 20 組模擬新歌情境（1 首原型 → 20 首假新歌）
PROFILES = [
    {'profile_name': '夏日高能量', 'loudness_delta': 4.0, 'tempo_scale': 1.15, 'duration_scale': 0.90, 'year_delta': 0, 'danceability_delta': 0.15, 'energy_delta': 0.20},
    {'profile_name': '抒情慢板', 'loudness_delta': -5.0, 'tempo_scale': 0.78, 'duration_scale': 1.12, 'year_delta': 0, 'danceability_delta': -0.05, 'energy_delta': -0.15},
    {'profile_name': '復古翻新', 'loudness_delta': -1.0, 'tempo_scale': 0.95, 'duration_scale': 1.0, 'year_delta': -18, 'danceability_delta': 0.0, 'energy_delta': 0.0},
    {'profile_name': '最新潮流', 'loudness_delta': 2.0, 'tempo_scale': 1.08, 'duration_scale': 0.94, 'year_delta': 10, 'danceability_delta': 0.12, 'energy_delta': 0.18},
    {'profile_name': '短歌衝刺', 'loudness_delta': 3.0, 'tempo_scale': 1.20, 'duration_scale': 0.72, 'year_delta': 2, 'danceability_delta': 0.18, 'energy_delta': 0.22},
    {'profile_name': '長曲敘事', 'loudness_delta': -2.0, 'tempo_scale': 0.88, 'duration_scale': 1.28, 'year_delta': 0, 'danceability_delta': -0.08, 'energy_delta': -0.10},
    {'profile_name': '舞曲場', 'loudness_delta': 5.0, 'tempo_scale': 1.25, 'duration_scale': 0.85, 'year_delta': 4, 'danceability_delta': 0.25, 'energy_delta': 0.28},
    {'profile_name': '民謠感', 'loudness_delta': -4.0, 'tempo_scale': 0.75, 'duration_scale': 1.06, 'year_delta': -6, 'danceability_delta': -0.10, 'energy_delta': -0.12},
    {'profile_name': '主流中庸', 'loudness_delta': 1.0, 'tempo_scale': 1.02, 'duration_scale': 1.0, 'year_delta': 1, 'danceability_delta': 0.05, 'energy_delta': 0.08},
    {'profile_name': '實驗邊緣', 'loudness_delta': -1.5, 'tempo_scale': 1.32, 'duration_scale': 0.68, 'year_delta': 6, 'danceability_delta': 0.10, 'energy_delta': 0.15},
    {'profile_name': '電音派對', 'loudness_delta': 6.0, 'tempo_scale': 1.28, 'duration_scale': 0.80, 'year_delta': 3, 'danceability_delta': 0.30, 'energy_delta': 0.35},
    {'profile_name': '原聲不插電', 'loudness_delta': -6.0, 'tempo_scale': 0.72, 'duration_scale': 1.15, 'year_delta': -3, 'danceability_delta': -0.15, 'energy_delta': -0.20},
    {'profile_name': '懷舊金曲', 'loudness_delta': 0.0, 'tempo_scale': 0.92, 'duration_scale': 1.05, 'year_delta': -12, 'danceability_delta': 0.02, 'energy_delta': 0.0},
    {'profile_name': '街頭嘻哈', 'loudness_delta': 2.5, 'tempo_scale': 1.10, 'duration_scale': 0.88, 'year_delta': 5, 'danceability_delta': 0.20, 'energy_delta': 0.25},
    {'profile_name': '電影配樂感', 'loudness_delta': -0.5, 'tempo_scale': 0.85, 'duration_scale': 1.35, 'year_delta': 0, 'danceability_delta': -0.05, 'energy_delta': 0.05},
    {'profile_name': '兒童節慶', 'loudness_delta': 1.5, 'tempo_scale': 1.18, 'duration_scale': 0.78, 'year_delta': 0, 'danceability_delta': 0.22, 'energy_delta': 0.18},
    {'profile_name': '金屬衝擊', 'loudness_delta': 5.5, 'tempo_scale': 1.35, 'duration_scale': 0.75, 'year_delta': 2, 'danceability_delta': 0.08, 'energy_delta': 0.40},
    {'profile_name': '爵士深夜', 'loudness_delta': -3.5, 'tempo_scale': 0.82, 'duration_scale': 1.20, 'year_delta': -8, 'danceability_delta': 0.10, 'energy_delta': -0.05},
    {'profile_name': '民謠清新', 'loudness_delta': -2.5, 'tempo_scale': 0.88, 'duration_scale': 1.02, 'year_delta': 4, 'danceability_delta': 0.05, 'energy_delta': 0.0},
    {'profile_name': '合成器流行', 'loudness_delta': 2.0, 'tempo_scale': 1.12, 'duration_scale': 0.92, 'year_delta': 7, 'danceability_delta': 0.15, 'energy_delta': 0.20},
]

# 三檔期商業假設（總預算 = 75+70+65 = 210）
DEFAULT_PERIODS = ['Q1_春夏檔', 'Q2_秋季檔', 'Q3_年末檔']
DEFAULT_BUDGET_BY_PERIOD = {'Q1_春夏檔': 75.0, 'Q2_秋季檔': 70.0, 'Q3_年末檔': 65.0}
DEFAULT_L_MIN_BY_PERIOD = {'Q1_春夏檔': 2, 'Q2_秋季檔': 2, 'Q3_年末檔': 1}
DEFAULT_U_MAX_BY_PERIOD = {'Q1_春夏檔': 4, 'Q2_秋季檔': 4, 'Q3_年末檔': 3}

# 聲學門檻初值（strategy 會依 20 首候選曲分位數自動校準後寫入）
DEFAULT_ACOUSTIC_POLICY = {
    'Q1_春夏檔': {'loudness': None, 'tempo': 105.0, 'danceability': None, 'energy': 0.15},
    'Q2_秋季檔': {'loudness': None, 'tempo': 95.0, 'danceability': None, 'energy': None},
    'Q3_年末檔': {'loudness': -6.0, 'tempo': 100.0, 'danceability': 0.10, 'energy': 0.20},
}
