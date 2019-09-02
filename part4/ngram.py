import re
import math
import sys
import codecs

PREFIX = '<s> '
SUFFIX = ' </s>'
UNKNOWN = '<unk>'

"""
n - size of n-grams
text - list of words/strings
returns: tuple of n-1 (word, context)
"""


def get_ngrams(n, text):  # - generator
    prefix = PREFIX
    suffix = SUFFIX
    for i in range(n - 2):
        prefix += PREFIX

    sentences = re.split(r'[?,.!:;]+', text)
    for sentence in sentences:
        sentence = re.sub(r'[^\w<>]', ' ', sentence)
        sentence = sentence.strip()
        sentence = prefix + sentence + suffix
        words = sentence.split()

        for i in range(n - 1, len(words)):
            context = ''
            for j in (1, n - 1):
                context = words[i - j] + ' ' + context
            yield words[i], context


"""
n - number of grams
corpus path - file path
returns: NGramLM trained model
"""


def create_ngramlm(n, corpus_path, delta=.0):
    ngramlm = NGramLM(n, delta)
    with codecs.open(corpus_path, 'r', 'utf-8') as file:
        corpus = file.read()
        corpus = mask_rare(corpus)
        ngramlm.update(corpus)

    return ngramlm


"""
n - number of grams
corpus path - file path
returns: NGramLM trained model
"""


def create_interpolator(n, lambdas, corpus_path):
    inter = NGramInterpolator(n, lambdas)
    with codecs.open(corpus_path, 'r', 'utf-8') as file:
        corpus = file.read()
        corpus = mask_rare(corpus)
        inter.update(corpus)

    return inter


"""
returns log(e) prob of text (list of words/strings)
under NGramLM trained model
"""


def text_prob(model, text, delta=.0):
    log_prob = 0
    ngrams = get_ngrams(model.n, text)
    for word, context in ngrams:
        prob = model.word_prob(word, context, delta)
        log_prob += math.log(prob)

    return log_prob


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
        count += 1
        all_words[word] = count

    for word, count in all_words.items():
        if count == 1:
            rare_words.add(word)
    print("all: ", len(all_words), ", rare: ", len(rare_words))

    masked_corpus = []
    sentences = corpus.split()
    for sentence in sentences:
        words = re.split(r'[^\w]', sentence)
        for word in words:
            if word in rare_words:
                sentence = sentence.replace(word, UNKNOWN)
        masked_corpus.append(sentence)
        masked_corpus.append(' ')

    return ''.join(masked_corpus).strip()


"""
model - trained model
corpus_path - test data file path, no need to mask rare words
returns perplexity of a trained model
counts N, total tokens
"""


def perplexity(model, corpus_path):
    with codecs.open(corpus_path, 'r', 'utf-8') as file:
        corpus = file.read()
        test_set_model = create_ngramlm(1, corpus_path)     # create an unigram model
        num_tokens = len(test_set_model.vocabulary)         # use the unigram model to get N
        prob = text_prob(model, corpus, model.delta) / num_tokens

        return prob


class NGramLM:
    def __init__(self, n, delta=.0):
        self.n = n  # size of n-grams
        self.ngram_counts = {}  # n-grams seen in the training data,
        self.context_counts = {}  # contexts seen in the training data,
        self.vocabulary = {}  # words seen in the traniing data
        self.delta = delta

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
            vocabulary_counter = self.vocabulary.get(word, 0)
            ngram_counter += 1
            context_counter += 1
            vocabulary_counter += 1
            self.ngram_counts[word_with_context] = ngram_counter
            self.context_counts[context] = context_counter
            self.vocabulary[word] = vocabulary_counter

        # print(self.vocabulary[UNKNOWN])

    """
    returns prob of n-gram (word, context) using internal counters.
    If context is previously unseen, returns 1/|V|, V is vocabulary model
    """

    def word_prob_delta_zero(self, word, context):
        word_with_context = get_word_with_context(word, context)
        ngram_counter = self.ngram_counts.get(word_with_context, 0)
        context_counter = self.context_counts.get(context, 0)
        if context_counter == 0 or word not in self.vocabulary:
            return self.vocabulary[UNKNOWN] / (len(self.vocabulary) * 1.0)
        else:
            return ngram_counter / (context_counter * 1.0)


    """
    prob with smoothing
    returns: Laplace-smoothed probabilities 
    (need to modify to fit n-gram)
    """

    def word_prob(self, word, context, delta=.0):
        if delta == 0:
            return self.word_prob_delta_zero(word, context)

        word_with_context = get_word_with_context(word, context)
        ngram_counter = self.ngram_counts.get(word_with_context, 0)
        context_counter = self.context_counts.get(context, 0)
        return (ngram_counter + delta) / ((context_counter + delta * len(self.vocabulary)) * 1.0)


"""
	linear interpolation.
"""

class NGramInterpolator:


    """
        n - size of the largest n-gram
        lambdas - a list of length n containing the interpolation factors
            (floats) in descending order of n-gram size.

        save n and lambdas
        initialize n internal NGramLMs, one for each n-gram size
    """

    def __init__(self, n, lambdas):
        self.n = n
        self.lambdas = lambdas
        self.models = []
        for i in range(n):
            self.models.append(NGramLM(n - i))


    """
        update all of the internal NGramLMs
    """

    def update(self, text):
        for model in self.models:
            model.update(text)


    """
        returns the linearly interpolated prob using lambdas
        and prob given by NGramLMs
    """

    def word_prob(self, word, context, delta=0):
        prob = 0
        for i in range(self.n):
            prob += self.lambdas[i] * self.models[i].word_prob(word, context, delta)

        return prob


def main(argv):
    model_smoothed_warpeace = create_ngramlm(3, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\warpeace.txt", 0.5)
    # model_warpeace = create_ngramlm(3, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\warpeace.txt")
    # print(perplexity(model_smoothed_warpeace, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\shakespeare.txt"))
    # print(perplexity(model_warpeace, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\shakespeare.txt"))

    model_smoothed_shakespeare = create_ngramlm(3, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\shakespeare.txt", 0.5)
    print(perplexity(model_smoothed_warpeace, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\sonnets.txt"))
    print(perplexity(model_smoothed_shakespeare, r"C:\Users\wenxi\OneDrive\UTD\nlp\hw1\sonnets.txt"))

    pass


if __name__ == '__main__':
    main(sys.argv)
