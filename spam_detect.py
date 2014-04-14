import numpy as np
import scipy as sp
import scipy.stats

wordmap = [line.strip().split(" ") for line in open("tmp/wordmap.txt")][1:]
wordmap = {word: int(num) for word, num in wordmap}
words = {num: word for word, num in wordmap.iteritems()}
words = [words[i] for i in range(len(words))]

word_features = np.loadtxt("tmp/model-final.phi")
word_features = word_features.T

spammy_words = [line.strip() for line in open("spam.txt")]
spammy_ids = [wordmap[word] for word in spammy_words]
nonspammy_ids = list(set(range(len(words))) - set(spammy_ids))
nonspammy_ids = np.random.choice(nonspammy_ids, len(nonspammy_ids) * 0.3, False)
undefined_ids = list(set(range(len(words))) - set(spammy_ids) - set(nonspammy_ids))

spammy_feats = word_features[spammy_ids]
nonspammy_feats = word_features[nonspammy_ids]
undefined_feats = word_features[undefined_ids]
feats_list = [spammy_feats, nonspammy_feats]

gamma = 0.01 # prior of spammy
Mu = np.zeros((word_features.shape[1], 2))
Sigma = np.zeros((word_features.shape[1], 2))

failure_feats = set()
for s in range(2):
    for i in range(word_features.shape[1]):
        Mu[i, s] = feats_list[s][:, i].mean()
        Sigma[i, s] = feats_list[s][:, i].std()
        if Sigma[i, s] == 0:
            failure_feats.add(i)
failure_feats = list(failure_feats)

for word_id in undefined_ids:
    probs = []
    feat = word_features[word_id]
    for s in range(2):
        if s == 0:
            prob_s = [gamma]
        else:
            prob_s = [1 - gamma]
        for i in range(word_features.shape[1]):
            if i in failure_feats:
                continue
            prob = sp.stats.norm.pdf(feat[i], Mu[i, s], Sigma[i, s])
            if np.isnan(prob):
                prob = 0
            prob_s.append(prob)
        probs.append(sp.misc.logsumexp(prob_s))
    if probs[0] > probs[1]:
        print "* %s" % words[word_id]
    else:
        print "%s" % words[word_id]

nonspam_topics = [2, 3, 6, 7, 9, 12, 16, 18, 19, 22, 23, 24, 25, 28, 30, 31, 33, 34, 35, 36, 37, 38, 40, 42, 43, 48, 50, 51, 54, 56, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 74, 77, 78, 80, 81, 82, 83, 86, 87, 89, 90, 92, 93, 97, 98]
spam_topics = [i for i in range(word_features.shape[1])
        if i not in nonspam_topics]


def logsum(*array):
    return np.log(array).sum()

def detect_spam(word, gamma=0.5):
    v = wordmap[word]
    feat = word_features[v]
    spam_feat = feat[spam_topics]
    p_spam = logsum([spam_feat.sum(), gamma])

    nonspam_feat = feat[nonspam_topics]
    p_nonspam = logsum(nonspam_feat.sum(), (1 - gamma))

    if p_spam > p_nonspam:
        return True
    else:
        return False

is_spam = []
for v, word in enumerate(words):
    is_spam = detect_spam(word, 0.2)
    if is_spam:
        print "* %s" % word
