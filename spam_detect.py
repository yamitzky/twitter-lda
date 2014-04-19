import numpy as np

class SpamDetector():
    def __init__(self, spam_topics, nonspam_topics):
        self.nonspam_topics = nonspam_topics
        self.spam_topics = spam_topics

    def detect(self, feature, gamma=0.1):
        spam_feat = feature[spam_topics]
        p_spam = self.logsum(spam_feat.sum(), gamma)

        nonspam_feat = feature[nonspam_topics]
        p_nonspam = self.logsum(nonspam_feat.sum(), 1 - gamma)

        return p_spam > p_nonspam

    def logsum(self, *array):
        return np.log(array).sum()

wordmap = [line.strip().split(" ") for line in open("tmp/wordmap.txt")][1:]
wordmap = {word: int(num) for word, num in wordmap}
words = {num: word for word, num in wordmap.iteritems()}
words = [words[i] for i in range(len(words))]

word_features = np.loadtxt("tmp/model-final.phi")
word_features = word_features.T

nonspam_topics = [int(line.strip()) for line in open("nonspam-topics.txt")]
spam_topics = [i for i in range(word_features.shape[1])
    if i not in nonspam_topics]

detector = SpamDetector(spam_topics, nonspam_topics)

for v, word in enumerate(words):
    v = wordmap[word]
    feature = word_features[v]

    is_spam = detector.detect(feature, 0.05)
    if is_spam:
        print "%s" % word
