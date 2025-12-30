import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def split_data(
    data_path: str,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load a dataset and split it into training and test sets.
    """
    # Load data
    df = pd.read_csv(data_path)

    # Separate features and label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def main(args):
    # ---------------------------------------------------------
    # TO DO: enable autologging
    # ---------------------------------------------------------
    mlflow.autolog()

    # Start MLflow run (optional but recommended for clarity)
    with mlflow.start_run():

        # Log split parameters explicitly
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("label_column", args.label_column)

        # Split the data
        X_train, X_test, y_train, y_test = split_data(
            data_path=args.data_path,
            label_column=args.label_column,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )
        model.fit(X_train, y_train)

        # Evaluation metrics are automatically logged by mlflow.autolog()
        model.score(X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--label-column", type=str, default="Label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)

    args = parser.parse_args()
    main(args)
