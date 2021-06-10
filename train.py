from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from dataset import get_dataset
from preprocess import pipeline
from word_senti import is_positive, is_negative

RANDOM_STATE = 2021

def split_train_test(df, random_state=RANDOM_STATE):
    df = shuffle(df, random_state=random_state)
    train = df[df.split == "train"]
    test = df[df.split == "test"]

    X_train = train.document.values
    y_train = train.label.map({"pos":1, "neg":-1}).values

    X_test = test.document.values
    y_test = test.label.map({"pos":1, "neg":-1}).values

    return X_train, X_test, y_train, y_test

def count_positive_words(tokens):
    return sum([1 for t in tokens if is_positive(t)])

def count_negative_words(tokens):
    return sum([1 for t in tokens if is_negative(t)])

def vectorize(words_tokens):
    counts = []
    for w in words_tokens:
        counts.append([count_positive_words(w), count_negative_words(w)])
    return counts

def vectorizer_pipeline(df, normalize=False):
    X_train, X_test, y_train, y_test = split_train_test(df)
    X_train_tok = pipeline(X_train)
    X_test_tok = pipeline(X_test)

    X_train_counts = vectorize(X_train_tok)
    X_test_counts = vectorize(X_test_tok)

    if normalize:
        normalizer = StandardScaler()
        X_train_counts = normalizer.fit_transform(X_train_counts)
        X_test_counts = normalizer.transform(X_test_counts)

    return X_train_counts, X_test_counts, y_train, y_test

def main():
    df = get_dataset()
    X_train, X_test, y_train, y_test = vectorizer_pipeline(df, normalize=True)
    clf = LogisticRegression(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    print("Intercept", clf.intercept_)
    print("Coef", clf.coef_)

    print("Acurácia de treino: ", clf.score(X_train, y_train))
    print("Acurácia de test: ", clf.score(X_test, y_test))


if __name__ == '__main__':
    main()