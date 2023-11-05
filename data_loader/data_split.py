from sklearn.model_selection import train_test_split


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_feature_and_labels(data):
    return data.iloc[:, :-3], data.iloc[:, -3:]
