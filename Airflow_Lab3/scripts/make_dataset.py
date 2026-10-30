"""Build the lab's CSV files from scikit-learn's built-in Wine dataset.

Run once (the outputs are committed, so you only need this to regenerate them):

    python scripts/make_dataset.py

Writes:
    dags/data/wine.csv         168 wines used to train the clustering model
    dags/data/new_samples.csv   10 held-out wines the DAG assigns to clusters at the end
                                (stands in for "new data arriving")

Both files keep the true cultivar in a `cultivar` column. The clustering never
sees it; it is only used afterwards to check how well the clusters match reality.
"""
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "dags" / "data"


def main() -> None:
    df = load_wine(as_frame=True).frame
    df = df.rename(columns={"target": "cultivar", "od280/od315_of_diluted_wines": "od280_od315"})

    train, new = train_test_split(df, test_size=10, stratify=df["cultivar"], random_state=7)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.sort_index().to_csv(DATA_DIR / "wine.csv", index=False)
    new.sort_index().to_csv(DATA_DIR / "new_samples.csv", index=False)
    print(f"wine.csv: {len(train)} rows, new_samples.csv: {len(new)} rows -> {DATA_DIR}")


if __name__ == "__main__":
    main()
