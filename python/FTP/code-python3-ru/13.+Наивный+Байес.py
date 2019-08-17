
# coding: utf-8

# In[2]:



# naive_bayes.py

import sys
sys.path.append("../code-python3-ru")

from collections import Counter, defaultdict
from lib.machine_learning import split_data
import math, random, re, glob

def tokenize(message):
    message = message.lower()                       # преобразовать с строчные
    all_words = re.findall("[a-z0-9']+", message)   # извлечь слова
    return set(all_words)                           # удалить повторы

def count_words(training_set):
    """обучающая выборка состоит из пар (сообщение, спам?"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """преобразовать частотности word_counts в список триплетов
    слово w, p(w | spam) и p(w | ~spam)"""
    return [(w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k))
             for w, (spam, non_spam) in counts.items()]

def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # если СЛОВО в сообщении появляется, то
        # добавить лог-вероятность встретить его в сообщении
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # если СЛОВО в сообщении не появляется, то
        # добавить лог-вероятность НЕ встретить его в сообщении,
        # вычисляемое как log(1 - вероятность встретить его в сообщении)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        # подсчитать спамные и неспамные сообщения
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        # пропустить обучающую выборку через "конвейер"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


def get_subject_data(path):

    data = []

    # регулярное выражение для удаления начального слова "Subject:" 
    # и любых пробельных символов после него
    subject_regex = re.compile(r"^Subject:\s+")

    # glob.glob возвращает имена файлов,
    # соответствующие шаблону поиска по указанному пути    
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        with open(fn,'r',encoding='ISO-8859-1') as file:
            for line in file:
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    data.append((subject, is_spam))

    return data

def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def train_and_test_model(path):

    data = get_subject_data(path)
    random.seed(0)      # чтобы получить повторимые ответы
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

    counts = Counter((is_spam, spam_probability > 0.5) # (фактическая, прогнозная)
                     for _, is_spam, spam_probability in classified)

    print(counts)

    classified.sort(key=lambda row: row[2])
    spammiest_hams = list(filter(lambda row: not row[1], classified))[-5:]
    hammiest_spams = list(filter(lambda row: row[1], classified))[:5]

    print("самые спамные среди не-спамных (spammiest_hams)", spammiest_hams)
    print()
    print("самые не-спамные среди спамных (hammiest_spams)", hammiest_spams)

    words = sorted(classifier.word_probs, key=p_spam_given_word)

    spammiest_words = words[-5:]
    hammiest_words = words[:5]
    
    print()
    print("самые спамные слова", spammiest_words)
    print()
    print("самые не-спамные слова", hammiest_words)


if __name__ == "__main__":
    train_and_test_model(r"../code-python3-ru/data/spam/*/*")
    #train_and_test_model(r"c:\spam\*\*")
    #train_and_test_model(r"/home/joel/src/spam/*/*")


# In[ ]:



