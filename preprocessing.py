def clean_data(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

# prepare features
def prepare_features(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    high_cardinality_cols = []

    for col in X.select_dtypes(include=['object', 'string']).columns:

        if X[col].nunique() > 50:
            high_cardinality_cols.append(col)

    print("\nDropped High Cardinality Columns:")
    print(high_cardinality_cols)

    X = X.drop(columns=high_cardinality_cols)

    X = pd.get_dummies(X, drop_first=True)

    return X, y

# detect problem type
from pandas.api.types import is_numeric_dtype

def detect_problem_type(y):

    print("\nDetected dtype:", y.dtype)

    if is_numeric_dtype(y):

        return "regression"

    else:

        return "classification"

# split data
def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
