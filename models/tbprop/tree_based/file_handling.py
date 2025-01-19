import pandas as pd
import matplotlib.pyplot as plt


def create_binary_label(df, ycol="AUC", threshold=1000):
    if ycol not in df.columns:
        raise ValueError(f"Column {ycol} not found in dataset")

    if ycol != "AUC":
        df.rename({ycol: "AUC"}, axis=1, inplace=True)
        ycol = "AUC"

    if df[ycol].max() == 1 and df[ycol].min() == 0:
        df[ycol + "_bin"] = df[ycol].copy()
    else:
        df[ycol + "_bin"] = df[ycol].apply(lambda x: 1 if x > threshold else 0)


def read_pk_dataset(data_path, mode="bin_class", ycol="AUC", verbose=True):
    df = pd.read_csv(data_path)

    if verbose:
        print(f"Shape of dataset = {df.shape}")
        print(f"Columns with null values = {list(df.isnull().sum()[df.isnull().sum() > 0].keys())}")

    id_cols = ["Cmpd Name", "mol"]
    remove_cols = set()

    remove_cols.update(id_cols)

    if mode == "bin_class":
        create_binary_label(df, ycol=ycol)

        label_cols = ["AUC", "AUC_bin"]
        remove_cols.update(label_cols)

        if verbose:
            print(f"Label distribution in train set:\n{df['AUC_bin'].value_counts(normalize=True)}")
    elif mode == "reg":
        label_cols = ["AUC"]
        remove_cols.update(label_cols)

    if verbose:
        df[ycol].hist(bins=20)
        plt.xlabel("AUC value")
        plt.title("Label (AUC) Distribution")
        plt.show()

    return df, remove_cols


def read_pk_dataset_with_split(
    train_path, test_path, val_path=None, mode="bin_class", ycol="AUC", verbose=True
):
    """
    Read PK dataset with train, test and validation splits.

    Parameters
    ----------
    train_path : str
        Path to train data file.
    test_path : str
        Path to test data file.
    val_path : str, optional
        Path to validation data file, by default None
    mode : str, optional
        Mode of dataset, by default 'bin_class'
    ycol : str, optional
        Name of label column, by default 'AUC'

    Returns
    -------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    df_val : pandas.DataFrame
        Validation data.
    remove_cols : set
        Columns to remove from dataset.
    """

    # Read data files
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Print data info
    if verbose:
        print(f"Shape of train data = {df_train.shape}")
        print(
            f"Columns with null values = {list(df_train.isnull().sum()[df_train.isnull().sum() > 0].keys())}"
        )

        print(f"Shape of test data = {df_test.shape}")
        print(
            f"Columns with null values = {list(df_test.isnull().sum()[df_test.isnull().sum() > 0].keys())}"
        )

    # Read validation data if present
    if val_path:
        df_val = pd.read_csv(val_path)
        if verbose:
            print(f"Shape of validation data = {df_val.shape}")
            print(
                f"Columns with null values = {list(df_val.isnull().sum()[df_val.isnull().sum() > 0].keys())}"
            )
    else:
        df_val = None

    id_cols = ["Cmpd Name", "mol"]
    remove_cols = set()

    remove_cols.update(id_cols)

    # Create binary label
    if mode == "bin_class":
        create_binary_label(df_train, ycol=ycol)
        create_binary_label(df_test, ycol=ycol)

        # Create binary label for validation set
        if df_val is not None:
            create_binary_label(df_val, ycol=ycol)

        label_cols = ["AUC", "AUC_bin"]
        remove_cols.update(label_cols)

        if verbose:
            print(
                f"Label distribution in train set:\n{df_train['AUC_bin'].value_counts(normalize=True)}"
            )
    elif mode == "reg":
        label_cols = ["AUC"]
        remove_cols.update(label_cols)

    if verbose:
        df_train["AUC"].hist(bins=20)
        plt.xlabel("AUC value")
        plt.title("Label (AUC) Distribution")
        plt.show()

    return df_train, df_val, df_test, remove_cols
