import math
from collections import defaultdict

# Tokenizes text into character n-grams; applies case folding
def tokenize(text, method='', n=3):
    if method == 'nbse':
        text = '<' + text + '>'
    text = text.lower()
    return [text[i:i+n] for i in range(len(text) - (n-1))]

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        return self.mfc

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'nb', or 'nbse'
    method = sys.argv[1]

    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]
    
    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]

    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        train_counts = count_vectorizer.fit_transform(train_texts)

        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)

        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        test_counts = count_vectorizer.transform(test_texts)
        # Predict the class for each test document
        results = clf.predict(test_counts)

    elif method == 'nb' or method == 'nbse':
        # Tokenize the training documents into character trigrams
        train_tokens = [tokenize(text, method) for text in train_texts]

        # Calculate class prior probabilities P(class)
        class_counts = {}
        for klass in train_klasses:
            class_counts[klass] = class_counts.get(klass, 0) + 1
        total_documents = len(train_texts)
        class_priors = {klass: count / total_documents for klass, count in class_counts.items()}

        # Count the number of occurences of each trigram per class and the total number of tokens per class
        trigram_counts = defaultdict(lambda: defaultdict(int))
        total_trigrams = {}
        for i, klass in enumerate(train_klasses):
            trigrams = train_tokens[i]
            total_trigrams[klass] = total_trigrams.get(klass, 0) + len(trigrams)
            for trigram in trigrams:
                trigram_counts[klass][trigram] += 1

        # Total number of trigram types in the training data
        types = set(trigram for trigram_list in trigram_counts.values() for trigram in trigram_list)
        V = len(types)

        # Check to make sure that all probabilities are between 0 and 1, and that the distributions sum to 1
        for klass in class_priors:
            one = 0
            for trigram in types:
                likelihood = (trigram_counts[klass][trigram] + 1) / (total_trigrams[klass] + V)
                if likelihood <= 0 or likelihood > 1:
                    print(f"Error: An exception occurred while calculating probabilities for class {klass}.")
                    sys.exit(1)
                one += likelihood
            if abs(one - 1) > 1e-10:
                print(f"Error: An exception occurred while calculating probabilities for class {klass}.")
                sys.exit(1)

        # Classify test documents
        results = []
        for text in test_texts:
            test_trigrams = tokenize(text, method)
            known_trigrams = [trigram for trigram in test_trigrams if trigram in types]
            class_scores = {}
            for klass in class_priors:
                # Calculate the log of class prior
                score = math.log(class_priors[klass])
                for trigram in known_trigrams:
                    # Calculate the log likelihood probabilities P(trigram|class) using add-1 smoothing
                    count = trigram_counts[klass][trigram] + 1
                    total = total_trigrams[klass] + V
                    score += math.log(count / total)
                class_scores[klass] = score
            # Find the class with the highest score
            predicted = max(class_scores, key=class_scores.get)
            results.append(predicted)

    for r in results:
        print(r)
