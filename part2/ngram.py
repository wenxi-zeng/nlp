import re
import math
import sys
import codecs

N_GRAM = 3

"""
n - size of n-grams
text - list of words/strings
returns: tuple of n-1 (word, context)
"""


def get_ngrams(n, text):  # - generator
    text = re.sub(r'[^\w]', ' ', text)
    prefix = '<s> '
    suffix = ' </s>'
    for i in range(n - 2):
        prefix += prefix
    text.strip()
    text = prefix + text + suffix
    words = text.split()

    for i in range(2, len(words)):
        yield words[i], words[i - 2] + ' ' + words[i - 1]


"""
n - number of grams
corpus path - file path
returns: NGramLM trained model
"""


def create_ngramlm(n, corpus_path):
    ngramlm = NGramLM(N_GRAM)
    with codecs.open(corpus_path, 'r', 'utf-8') as file:
        corpus = file.read()
        corpus = mask_rare(corpus)
        sentences = re.split(r'[?,.!:;]+', corpus)
        for sentence in sentences:
            # print(sentence)
            ngramlm.update(sentence)

    return ngramlm


"""
returns log(e) prob of text (list of words/strings)
under NGramLM trained model
"""


def text_prob(model, text):
    log_prob = 0
    ngrams = get_ngrams(N_GRAM, text)
    for word, context in ngrams:
        prob = model.word_prob(word, context)
        log_prob += math.log(prob)

    return math.exp(log_prob)


"""
concatenate word with context
"""


def get_word_with_context(word, context):
    return context + ' ' + word


"""
corpus - content of file
returns: a copy of corpus with words 
that appear only once replaced by the `<unk>' token
"""


def mask_rare(corpus):
    all_words = {}
    rare_words = set()
    words = re.split(r'[^\w]', corpus)
    for word in words:
        count = all_words.get(word, 0)
        all_words[word] = ++count

    for word, count in all_words.items():
        if count == 1:
            rare_words.add(word)

    for word in rare_words:
        corpus.replace(word, '<unk>')

    return corpus


class NGramLM:
    def __init__(self, n):
        self.n = n  # size of n-grams
        self.ngram_counts = {}  # n-grams seen in the training data,
        self.context_counts = {}  # contexts seen in the training data,
        self.vocabulary = set()  # words seen in the traniing data

    """
    updates NGramLM's internal counts
    and vocabulary for ngrams in text (list of words/strings)
    """

    def update(self, text):
        ngrams = get_ngrams(self.n, text)
        for word, context in ngrams:
            word_with_context = get_word_with_context(word, context)
            ngram_counter = self.ngram_counts.get(word_with_context, 0)
            context_counter = self.context_counts.get(context, 0)
            self.ngram_counts[word] = ++ngram_counter
            self.context_counts[context] = ++context_counter
            self.vocabulary.add(word)

    """
    returns prob of n-gram (word, context) using internal counters.
    If context is previously unseen, returns 1/|V|, V is vocabulary model
    """

    def word_prob(self, word, context):
        word_with_context = get_word_with_context(word, context)
        ngram_counter = self.ngram_counts.get(word_with_context, 0)
        context_counter = self.context_counts.get(context, 0)
        if context_counter == 0:
            return 1 / (len(self.vocabulary) * 1.0)
        else:
            return ngram_counter / (context_counter * 1.0)


def main(argv):
    model = create_ngramlm(3, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\warpeace.txt")
    # print(text_prob(model, "God has given it to me, let him who touches it beware!"))
    print(text_prob(model, "Where is the prince, my Dauphin?"))
    pass


if __name__ == '__main__':
    main(sys.argv)
