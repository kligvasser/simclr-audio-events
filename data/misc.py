import os
import pandas as pd
import librosa


AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac"]


def load_audio(audio_path, sample_rate=8000):
    (audio, sr) = librosa.load(audio_path, sr=None, mono=True)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def get_file_list(path_dir, max_size=None, extentions=AUDIO_EXTENSIONS):
    paths = list()

    for dirpath, _, files in os.walk(path_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(tuple(extentions)):
                paths.append(fname)

    return sorted(paths)[:max_size]


def save_df(df, df_path, index=False):
    ext = df_path.split('.')[-1]

    if ext == 'csv':
        df.to_csv(df_path, index=index)
    elif ext == 'xlsx':
        df.to_excel(df_path, index=index)
    elif ext == 'json':
        df.to_json(df_path, orient='records')
    elif ext == 'parquet':
        df.to_parquet(df_path, index=index)
    elif ext == 'pkl':
        df.to_pickle(df_path)
    else:
        raise ValueError(
            "Unsupported file format. Please choose from 'csv', 'xlsx', 'json', 'parquet', or 'pkl'."
        )


def load_df(df_path):
    ext = df_path.split('.')[-1]

    if ext == 'csv':
        df = pd.read_csv(df_path)
    elif ext == 'xlsx':
        df = pd.read_excel(df_path)
    elif ext == 'json':
        df = pd.read_json(df_path)
    elif ext == 'parquet':
        df = pd.read_parquet(df_path)
    elif ext == 'pkl':
        df = pd.read_pickle(df_path)
    else:
        raise ValueError(
            "Unsupported file format. Please choose from 'csv', 'xlsx', 'json', 'parquet', or 'pkl'."
        )

    return df
