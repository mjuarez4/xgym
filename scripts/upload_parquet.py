import os
import glob
import pandas as pd
from datasets import Dataset, Features, Value, Image
from tqdm import tqdm
from dataclasses import dataclass
import tyro

@dataclass
class Config:
    dir: str
    repo_id: str
    data_type: str  # "human" or "robot"
    batch_size: int = 50

def main(cfg: Config):
    data_dir = os.path.join(os.path.expanduser(cfg.dir), "data/chunk-000")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected directory not found: {data_dir}")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.parquet")))
    print(f"Found {len(parquet_files)} files in {data_dir}")


    # Define which image fields to treat as images
    image_fields = ["observation.image.low", "observation.image.wrist"]
    if cfg.data_type== "robot":
        image_fields.append("observation.image.side")

    for i in range(0, len(parquet_files), cfg.batch_size):
        batch_files = parquet_files[i:i + cfg.batch_size]
        dfs = []

        print(f"\nProcessing batch {i // cfg.batch_size} ({len(batch_files)} files)")
        for file_path in tqdm(batch_files):
            try:
                df = pd.read_parquet(file_path)

                for col in df.columns:
                    if col not in image_fields:
                        df[col] = df[col].apply(lambda x: str(x)[:1000])

                dfs.append(df)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        if not dfs:
            print(f"Skipping batch {i // cfg.batch_size} — no valid data.")
            continue

        df_batch = pd.concat(dfs, ignore_index=True)

        features = Features({
            col: Image(decode=True) if col in image_fields else Value("string")
            for col in df_batch.columns
        })

        try:
            dataset = Dataset.from_pandas(df_batch, features=features)
            dataset.push_to_hub(cfg.repo_id, split=f"batch_{i // cfg.batch_size}")
            print(f"Uploaded batch {i // cfg.batch_size} to {cfg.repo_id}")
        except Exception as e:
            print(f"Push failed for batch {i // cfg.batch_size}: {e}")

if __name__ == "__main__":
    tyro.cli(main)

