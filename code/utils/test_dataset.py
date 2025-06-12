import pytest
import pandas as pd


from utils.dataset import (
    retain_only_folds,
    split_df_by_folds,
    convert_shortening_units_to_num,
)


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "fold": [0, 0, 1, 1, 2, 2, 3, 3],
            "name": [f"sample_{i}" for i in range(8)],
        }
    )


@pytest.mark.parametrize(
    ("units_str", "data_size", "expected"),
    [
        ("50%", 100, 50),
        ("33%", 9, 2),  # floor via int()
        ("10u", 1_000, 10),
        (None, 123, None),
    ],
)
def test_convert_shortening_units_to_num_valid(units_str, data_size, expected):
    assert convert_shortening_units_to_num(units_str, data_size) == expected


@pytest.mark.parametrize("units_str", ["42", "abc", "50x"])
def test_convert_shortening_units_to_num_invalid(units_str):
    with pytest.raises(ValueError):
        convert_shortening_units_to_num(units_str, 10)


def test_retain_only_folds_subset(simple_df):
    out = retain_only_folds(simple_df, folds=[0, 2])
    assert set(out["fold"].unique()) == {0, 2}
    assert len(out) == 4


def test_retain_only_folds_none_keeps_all(simple_df):
    out = retain_only_folds(simple_df, folds=None)
    pd.testing.assert_frame_equal(out, simple_df)


def test_retain_only_folds_shortening_units(simple_df):
    # Two rows with fold 0 → shorten to just 1 row
    out = retain_only_folds(simple_df, folds=[0], shorten_string="1u", seed=123)
    assert len(out) == 1
    assert out.iloc[0]["fold"] == 0


def test_retain_only_folds_bad_fold_id(simple_df):
    with pytest.raises(AssertionError):
        retain_only_folds(simple_df, folds=[99])


def test_retain_only_folds_non_integer_fold(simple_df):
    with pytest.raises(AssertionError):
        retain_only_folds(simple_df, folds=[0, "1"])


def test_split_df_by_folds_auto_train(simple_df):
    train, eval_ = split_df_by_folds(simple_df, train_folds=None, eval_folds=[0])
    assert set(eval_["fold"].unique()) == {0}
    assert set(train["fold"].unique()) == {1, 2, 3}
    # No overlap and all rows accounted for
    assert len(train) + len(eval_) == len(simple_df)


def test_split_df_by_folds_explicit(simple_df):
    train, eval_ = split_df_by_folds(simple_df, train_folds=[1, 3], eval_folds=[0, 2])
    assert set(train["fold"].unique()) == {1, 3}
    assert set(eval_["fold"].unique()) == {0, 2}


def test_split_df_by_folds_overlap_raises(simple_df):
    with pytest.raises(AssertionError):
        split_df_by_folds(simple_df, train_folds=[0, 1], eval_folds=[1, 2])
