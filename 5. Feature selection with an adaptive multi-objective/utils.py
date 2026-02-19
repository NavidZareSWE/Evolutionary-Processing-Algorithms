import numpy as np

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_dataset(filepath):
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=0)
        if np.isnan(data).any():
            data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y
    except Exception as e:
        raise ValueError("Error loading dataset: {}".format(e))


def normalize_data(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1  # Avoid division by zero
    return (X - X_min) / denom


def train_test_split(X, y, train_ratio=0.7, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def format_pareto_front(pf):
    if not pf:
        return "[]"
    formatted = []
    for point in pf:
        error = float(point[0])
        n_feat = int(point[1])
        formatted.append("[{:.2f}, {}]".format(error, n_feat))
    return "[" + ", ".join(formatted) + "]"


def convert_pareto_front(pf):
    if not pf:
        return []
    return [[float(point[0]), int(point[1])] for point in pf]


def knn_predict(X_train, y_train, X_test, k=3):
    # Compute squared distances (skip sqrt of Euclidean distance since we only need relative ordering)
    # Using the formula: ||a-b||² = ||a||² + ||b||² - 2·(a·b)
    train_sq = np.sum(X_train ** 2, axis=1)
    test_sq = np.sum(X_test ** 2, axis=1)
    cross_term = np.dot(X_test, X_train.T)
    # Simply put, numpy.newaxis is used to increase the dimension of the existing array by one more dimension, when used once.
    # Read More: https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis#comment114074218_41267079
    distances_sq = test_sq[:, np.newaxis] + \
        train_sq[np.newaxis, :] - 2 * cross_term

    # argpartition is faster than full sorting when we only need the k smallest values
    # It rearranges indices so that the k smallest elements come first (but not necessarily in sorted order)
    # Read More: https://stackoverflow.com/a/52465229/27639316
    k_nearest_indices = np.argpartition(distances_sq, k, axis=1)[:, :k]
    k_nearest_labels = y_train[k_nearest_indices]

    # Majority voting
    predictions = np.zeros(X_test.shape[0], dtype=y_train.dtype)
    for i in range(X_test.shape[0]):
        unique, counts = np.unique(k_nearest_labels[i], return_counts=True)
        predictions[i] = unique[np.argmax(counts)]

    return predictions


def cross_validation_error(X, y, selected_features, n_folds=3, k=3):
    total_features = X.shape[1]
    num_feats_selected = np.sum(selected_features)

    if num_feats_selected == 0:
        return 100.0
    elif num_feats_selected == total_features:
        X_selected = X
    else:
        feature_indices = np.where(selected_features == 1)[0]
        X_selected = X[:, feature_indices]

    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)

    total_errors = 0
    total_samples = 0

    for fold in range(n_folds):
        test_start = fold * fold_size
        if fold == n_folds - 1:
            test_end = n_samples
        else:
            test_end = (fold + 1) * fold_size

        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate(
            [indices[:test_start], indices[test_end:]])

        X_train, y_train = X_selected[train_indices], y[train_indices]
        X_test, y_test = X_selected[test_indices], y[test_indices]

        predictions = knn_predict(X_train, y_train, X_test, k)
        errors = np.sum(predictions != y_test)

        total_errors += errors
        total_samples += len(y_test)

    error_rate = (total_errors / total_samples) * 100.0
    return error_rate


def compute_all_features_error(X, y, n_folds=3, k=3):
    all_features = np.ones(X.shape[1], dtype=int)
    return cross_validation_error(X, y, all_features, n_folds, k)
