import codecs


def read_words_as_dict(nome_arquivo):
    with codecs.open(nome_arquivo, encoding="latin1") as f:
        words = [line.strip() for line in f if line.strip() and not line.startswith(";")]

    return {w: None for w in words}


neg_words = read_words_as_dict("resources/negative-words.txt")
pos_words = read_words_as_dict("resources/positive-words.txt")


def is_negative(word):
    return word in neg_words


def is_positive(word):
    return word in pos_words

if __name__ == '__main__':
    print(neg_words)
    print(is_positive("abnormal"))
