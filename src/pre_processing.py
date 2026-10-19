import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CyberAttackPreprocessor:
    """
    Prepares the dataset for classification.

    Responsibilities:
    - Split the dataset into training and test partitions.
    - Remove excluded features.
    - Handle missing service values.
    - One-hot encode categorical features.
    - Scale numerical features when required.
    - Apply fitted transformations consistently to test data.
    """

    TARGET = "attack_cat"

    FEATURES_TO_REMOVE = ["id", "ct_ftp_cmd", "sloss", "dloss"]

    NORMINAL_FEATURES = ["proto", "service", "state", "is_ftp_login"]

    def __init__(self, dataset: pd.DataFrame):
        """
        Initialises the preprocessor with the original dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset containing predictors and the attack_cat target.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")

        if self.TARGET not in dataset.columns:
            raise ValueError(
                f"The dataset must contain the target column "
                f"'{self.TARGET}'."
            )

        self.dataset = dataset.copy()

        # Store the results of the train/test split.
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Store fitted preprocessing transformers separately for each model.
        self.preprocessors = {}

    def split_data(self, test_size: float = 0.20, random_state: int = 42):
        """
        Splits the dataset into training and test partitions.

        Stratification preserves the approximate class proportions
        in the two partitions. (Addresses substantial class imbalance)

        Returns
        -------
        X_train, X_test, y_train, y_test
            Training and test predictors and target labels.
        """
        X = self.dataset.drop(columns=[self.TARGET])
        y = self.dataset[self.TARGET]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        )

        return (self.X_train, self.X_test, self.y_train, self.y_test)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature removal and missing-value handling.

        This method does not fit any statistical transformations.
        """
        X = X.copy()

        X = X.drop(
            columns=self.FEATURES_TO_REMOVE,
            errors="ignore",
        )

        # Represents missing service values as an explicit category.
        if "service" in X.columns:
            X["service"] = (
                X["service"]
                .astype("object")
                .fillna("__MISSING__")
            )

        # Treats the specified categorical variables as categorical.
        for column in self.NORMINAL_FEATURES:
            if column in X.columns:
                X[column] = X[column].astype("object")

        return X

    def preprocess_data(self, X_train: pd.DataFrame, model_type: str = "knn") -> pd.DataFrame:
        """
        Fits preprocessing transformations on training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training predictors only. Do not include attack_cat.
        model_type : str
            Either 'knn' or 'tree'.

            'knn': Standardise numerical features.
            'tree': Leave numerical features unscaled.

        Returns
        -------
        pd.DataFrame
            Fully preprocessed training predictors.
        """
        model_type = model_type.lower()

        if model_type not in {"knn", "tree"}:
            raise ValueError(
                "model_type must be either 'knn' or 'tree'."
            )

        if self.TARGET in X_train.columns:
            raise ValueError(f"'{self.TARGET}' must not be included in X_train.")

        X = self._prepare_features(X_train)

        categorical_columns = [column for column in self.NORMINAL_FEATURES if column in X.columns]

        numerical_columns = [column for column in X.columns if column not in categorical_columns]

        # Constructs the numerical transformation.
        if model_type == "knn":
            numerical_transformer = StandardScaler()
        else:
            numerical_transformer = "passthrough"

        # Constructs the preprocessing pipeline.
        transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                ),
                (
                    "numerical",
                    numerical_transformer,
                    numerical_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # Fits exclusively on the training partition.
        processed_array = transformer.fit_transform(X)

        # Retains the fitted transformer for future test-data transformations.
        self.preprocessors[model_type] = transformer

        feature_names = transformer.get_feature_names_out()

        return pd.DataFrame(processed_array, columns=feature_names, index=X.index)

    def transform_test(
        self,
        X_test: pd.DataFrame,
        model_type: str = "knn",
    ) -> pd.DataFrame:
        """
        Transforms test predictors using the fitted training transformer.

        This method never refits the encoder or scaler on test data.
        """
        model_type = model_type.lower()

        if model_type not in self.preprocessors:
            raise ValueError(
                f"No fitted preprocessor exists for '{model_type}'. "
                "Call preprocess_data() on the training data first."
            )

        if self.TARGET in X_test.columns:
            raise ValueError(
                f"'{self.TARGET}' must not be included in X_test."
            )

        X = self._prepare_features(X_test)

        transformer = self.preprocessors[model_type]

        processed_array = transformer.transform(X)

        feature_names = transformer.get_feature_names_out()

        return pd.DataFrame(
            processed_array,
            columns=feature_names,
            index=X.index,
        )