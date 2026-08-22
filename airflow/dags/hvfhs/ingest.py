"""Descarga y muestreo del parquet público de NYC TLC."""
import pyarrow as pa
import pyarrow.parquet as pq

from hvfhs.constants import RAW_COLUMNS, SAMPLE_SIZE, RANDOM_STATE, TLC_URL


def download_sample(local_path: str = "/tmp/fhvhv.parquet", n: int = SAMPLE_SIZE):
    """Descarga el parquet, lee por row groups y devuelve una muestra de n filas."""
    import urllib.request

    urllib.request.urlretrieve(TLC_URL, local_path)

    pf = pq.ParquetFile(local_path)
    frames, total = [], 0
    for rg in range(pf.num_row_groups):
        frames.append(pf.read_row_group(rg, columns=RAW_COLUMNS))
        total += frames[-1].num_rows
        if total >= n:
            break

    df = pa.concat_tables(frames).to_pandas()
    return df.sample(n=min(n, df.shape[0]), random_state=RANDOM_STATE).reset_index(drop=True)
