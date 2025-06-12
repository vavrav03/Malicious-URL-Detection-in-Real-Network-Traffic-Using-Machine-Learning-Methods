"""Everything that has to do with loading data (from private or from public datasets) should be in this file"""

import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import mlflow

# malware and malicious are never in same dataset
TEXT_TO_NUMERIC_MAPPING = {"benign": 0, "malicious": 1, "phishing": 2, "malware": 1}
NUMERIC_TO_TEXT_MAPPING = {0: "benign", 1: "malicious"}
NUMERIC_TO_TEXT_MAPPING_KAGGLE_MULTIPLE = {0: "benign", 1: "malware", 2: "phishing"}



class URLDatasetPart(torch.utils.data.Dataset):
    """Training, validation, testing part of the dataset"""

    urls: list
    labels: list

    def __init__(self, urls, y):
        self.urls = list(urls) if not isinstance(urls, list) else urls
        self.labels = list(y) if not isinstance(y, list) else y

        assert len(self.urls) == len(self.labels), "Mismatched lengths in dataset part"

    @classmethod
    def from_pandas(cls, df):
        return cls(
            urls=df["url"],
            y=df["label"],
        )

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, i):
        return self.urls[i], self.labels[i]

    def log_to_mlflow(self, part_name, source_name, should_save: bool = True):
        """
        Log this dataset part to MLflow under the given name and context (e.g., 'train', 'val', 'test').
        """
        df = pd.DataFrame({"url": self.urls, "label": self.labels})
        mlflow.log_input(
            mlflow.data.from_pandas(df, name=part_name, source=source_name),
            context=source_name,
        )
        if should_save:
            mlflow.log_text(df.to_csv(index=False), artifact_file=f"{part_name}.csv")


def load_full_private_df(spark, table_name=None):
    pass


def load_public_dataset(df_name):
    dataset_path = os.getenv(f"DATASET_PATH_{df_name}")
    source_train = os.path.join(dataset_path, "train.csv")
    source_test = os.path.join(dataset_path, "test.csv")
    assert dataset_path is not None, f"No path has been found for this dataset {dataset_path}"
    print(f"[dataset] Using public dataset {df_name} with path: {dataset_path}")
    df_train = pd.read_csv(source_train)
    df_test = pd.read_csv(source_test)
    df_train["label"] = df_train["label"].map(TEXT_TO_NUMERIC_MAPPING)
    df_test["label"] = df_test["label"].map(TEXT_TO_NUMERIC_MAPPING)
    return df_train, df_test


def shorten_df_in_smart_way(df, target_size, seed=42):
    """
    - uniformly sample from folds
    - retain label ratio in folds form before shortening
    """
    if not {"url", "fold", "label"}.issubset(df.columns):
        raise ValueError("df must include 'url', 'fold', 'label' columns")
    folds = df["fold"].unique()
    quota = int(target_size / len(folds))
    parts = []

    for fold in folds:
        df_fold = df[df["fold"] == fold]
        if pd.isna(fold):
            # this does not match via == fold
            df_fold = df[df["fold"].isna()]
        else:
            df_fold = df[df["fold"] == fold]
        stratify_col = df_fold["label"] if df_fold["label"].nunique() > 1 else None
        try:
            sampled, _ = train_test_split(
                df_fold,
                train_size=quota,
                stratify=stratify_col,
                random_state=seed,
            )
        except ValueError:
            sampled = df_fold.sample(
                quota, replace=len(df_fold) < quota, random_state=seed
            )

        parts.append(sampled)

    return pd.concat(parts, ignore_index=True).reset_index(drop=True)

def retain_only_folds(df, folds, shorten_string=None, seed=None):
    if folds is not None:
        all_folds = sorted(df["fold"].unique())
        print(folds, all_folds)
        all_folds_clean = [f for f in all_folds if pd.notna(f)]
        folds_clean = [f for f in folds if pd.notna(f)]
        assert set(folds_clean).issubset(all_folds_clean), f"folds {folds_clean} must be subset of all_folds {all_folds_clean}"
        df = df.loc[df["fold"].isin(folds)]
    else:
        print("[dataset] Folds are none, no selection based on folds is applied")
    shorten_num = convert_shortening_units_to_num(shorten_string, len(df))
    if shorten_num:
        print(f"[dataset] Shortening applied to df of length: {len(df)} by ({shorten_string})")
        df = shorten_df_in_smart_way(df, shorten_num)
        print(f"[dataset] New length is {len(df)}")
    return df


def split_df_by_folds(df, train_folds, eval_folds, shorten_string_train=None, shorten_string_eval=None, seed=None):
    assert eval_folds is not None
    all_folds = sorted(df["fold"].unique())
    print(f"[dataset]: All folds {all_folds}")
    if train_folds is None:
        train_folds = [f for f in all_folds if f not in eval_folds]
    assert set(train_folds).isdisjoint(set(eval_folds)), "train_folds and eval_folds must not intersect"
    df_train = retain_only_folds(df, train_folds, shorten_string_train, seed)
    df_eval = retain_only_folds(df, eval_folds, shorten_string_eval, seed)
    print(f"[dataset]: train length: {len(df_train)}, eval_length: {len(df_eval)}")

    return df_train, df_eval


def convert_shortening_units_to_num(units_str, data_size):
    """Units must either end with '%' or 'u'"""
    if units_str == None:
        return None
    if units_str[-1] == "%":
        return int(float(units_str[:-1]) / 100 * data_size)
    elif units_str[-1] == "u":
        return int(units_str[:-1])
    else:
        raise ValueError("Invalid dataset shortenation units")


def get_df_from_args(args, spark):
    if args.dataset_name == "kaggle_multiple":
        args.label_count = 3
    else:
        args.label_count = 2

    if args.dataset_name == "private_data":
        df = load_full_private_df(spark)
        df_train, df_eval = split_df_by_folds(
            df,
            train_folds=args.train_folds,
            eval_folds=args.eval_folds,
            shorten_string_train=args.shorten_to_train,
            shorten_string_eval=args.shorten_to_eval,
            seed=args.seed,
        )
    else:
        df_train, df_eval = load_public_dataset(args.dataset_name)
        df_train = retain_only_folds(df_train, args.train_folds, args.shorten_to_train)
        df_eval = retain_only_folds(df_eval, args.eval_folds, args.shorten_to_eval)
    return df_train, df_eval


def get_dataset_from_args(args, spark):
    df_train, df_eval = get_df_from_args(args, spark)
    train_dataset = URLDatasetPart.from_pandas(df_train)
    eval_dataset = URLDatasetPart.from_pandas(df_eval)
    return train_dataset, eval_dataset

def get_df_by_folds_from_args(args, spark):
    if args.eval_folds is None:
        return get_df_from_args(args, spark)
    if args.dataset_name == "kaggle_multiple":
        args.label_count = 3
    else:
        args.label_count = 2

    if args.dataset_name == "private_data":
        df = load_full_private_df(spark)
    else:
        df_train, df_test = load_public_dataset(args.dataset_name)
        df = pd.concat([df_train, df_test], ignore_index=True)

    df_train, df_eval = split_df_by_folds(
        df,
        train_folds=args.train_folds,
        eval_folds=args.eval_folds,
        shorten_string_train=args.shorten_to_train,
        shorten_string_eval=args.shorten_to_eval,
        seed=args.seed,
    )
    return df_train, df_eval

def get_validation_dataset_df(args, spark):
    if args.dataset_name == "kaggle_multiple":
        args.label_count = 3
    else:
        args.label_count = 2

    if args.dataset_name == "private_data":
        df = load_full_private_df(spark)
    else:
        df, _ = load_public_dataset(args.dataset_name)

    df_train, df_eval = split_df_by_folds(
        df,
        train_folds=args.train_folds,
        eval_folds=args.eval_folds,
        shorten_string_train=args.shorten_to_train,
        shorten_string_eval=args.shorten_to_eval,
        seed=args.seed,
    )
    return df_train, df_eval

def get_validation_dataset_by_args(args, spark):
    df_train, df_eval = get_validation_dataset_df(args, spark)
    train_dataset = URLDatasetPart.from_pandas(df_train)
    eval_dataset = URLDatasetPart.from_pandas(df_eval)
    return train_dataset, eval_dataset


def analyze_folds(df: pd.DataFrame, top_k: int = 3):
    """
    Dynamically summarize each fold, including:
    - total sample count
    - count and % of each class in 'label'
    - number of unique SLDs
    - top-K domains by frequency (as separate columns)
    """
    all_labels = sorted(df["label"].unique())
    rows = []

    for fold, g in df.groupby("fold", sort=True):
        row = {
            "fold": fold,
            "total": len(g),
            "unique_domains": g["sld"].nunique(),
        }

        label_counts = g["label"].value_counts()
        for label in all_labels:
            count = label_counts.get(label, 0)
            row[label] = count
            row[f"{label}_pct"] = count / len(g) * 100

        top = g["sld"].value_counts(normalize=True).head(top_k).mul(100).round(1)
        tops = [f"{domain} ({pct}%)" for domain, pct in top.items()]
        tops += [""] * (top_k - len(tops))
        for i, val in enumerate(tops, start=1):
            row[f"top_{i}"] = val

        rows.append(row)

    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def get_domain_stats(df: pd.DataFrame):
    domain_stats = (
        df.groupby("sld")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={"benign": "benign", "malicious": "malicious"})
        .assign(total_records=lambda x: x["benign"] + x["malicious"])
        .reset_index()
    )
    domain_stats["malicious_ratio"] = domain_stats["malicious"] / domain_stats["total_records"]
    domain_stats = domain_stats.sort_values(by="total_records", ascending=False).reset_index(drop=True)
    return domain_stats