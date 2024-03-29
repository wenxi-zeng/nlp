import re
import math
import sys
import codecs
import random

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
            for j in range(i - n + 1, i):
                context = context + ' ' + words[j].strip()
            yield words[i].strip(), context.strip()


"""
n - number of grams
corpus path - file path
returns: NGramLM trained model
"""


def create_ngramlm(n, corpus_path, delta=.0):
    ngramlm = NGramLM(n, delta)
    with codecs.open(corpus_path, 'r', 'utf-8') as file:
        corpus = file.read()
        # corpus = mask_rare(corpus)
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
    return context.strip() + ' ' + word.strip()


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
        num_tokens = 0

        sentences = re.split(r'[?,.!:;]+', corpus)
        for sentence in sentences:
            words = re.split(r'[^\w]', sentence)
            num_tokens += len(words)

        i = text_prob(model, corpus, model.delta) / num_tokens
        return 2 ** -i


def random_text(model, max_length, delta=0):
    context = []
    suffix = SUFFIX.strip()
    for i in range(model.n - 1):
        context.append(PREFIX.strip())
    counter = 0
    sentence = ''

    while True:
        word = model.random_word(' '.join(context).strip(), delta)
        if word == suffix:
            return sentence.strip()

        sentence = sentence + ' ' + word
        counter += 1
        if word == suffix or counter > max_length:
            return sentence.strip()

        for i in range(len(context) - 1):
            context[i] = context[i + 1]
        context[len(context) - 1] = word


"""
almost identical to random_text
"""


def likeliest_text(model, max_length, delta = 0):
    context = []
    suffix = SUFFIX.strip()
    for i in range(model.n - 1):
        context.append(PREFIX.strip())
    counter = 0
    sentence = ''

    while True:
        word = model.likeliest_word(' '.join(context).strip(), delta)
        if word == suffix:
            return sentence.strip()

        sentence = sentence + ' ' + word
        counter += 1
        if word == suffix or counter > max_length:
            return sentence.strip()

        for i in range(len(context) - 1):
            context[i] = context[i + 1]
        context[len(context) - 1] = word


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
    returns a word sampled from the model's 
    probability distribution for context
    """

    def random_word(self, context, delta=0):
        vocabulary = list(self.vocabulary.keys())
        vocabulary = sorted(vocabulary)

        distrib = []
        total_counts = 0
        for word in vocabulary:
            word_with_context = get_word_with_context(word, context)
            count = self.ngram_counts.get(word_with_context, 0)
            total_counts += count
            distrib.append(total_counts)

        r = random.random()
        for i in range(len(distrib)):
            distrib[i] = distrib[i] / (total_counts * 1.0)
            if r < distrib[i]:
                return vocabulary[i]

        return vocabulary[len(vocabulary) - 1]


    """
        returns n-gram with the highest prob for context
    """

    def likeliest_word(self, context, delta=0):
        vocabulary = list(self.vocabulary.keys())

        distrib = []
        for word in vocabulary:
            word_with_context = get_word_with_context(word, context)
            count = self.ngram_counts.get(word_with_context, 0)
            distrib.append(count)

        max = 0
        maxIndex = 0
        for i in range(len(distrib)):
            if max < distrib[i]:
                max = distrib[i]
                maxIndex = i

        return vocabulary[maxIndex]

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
    random.seed(1)
    model = create_ngramlm(3, "shakespeare.txt")
    print(random_text(model, 10))
    print(random_text(model, 10))
    print(random_text(model, 10))
    print(random_text(model, 10))
    print(random_text(model, 10))

    bimodel = create_ngramlm(2, "shakespeare.txt")
    print(likeliest_text(bimodel, 10))
    trimodel = create_ngramlm(3, "shakespeare.txt")
    print(likeliest_text(trimodel, 10))
    quadmodel = create_ngramlm(4, "shakespeare.txt")
    print(likeliest_text(quadmodel, 10))
    pentamodel = create_ngramlm(5, "shakespeare.txt")
    print(likeliest_text(pentamodel, 10))

    pass


if __name__ == '__main__':
    main(sys.argv)
