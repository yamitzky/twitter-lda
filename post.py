import csv

import numpy as np

# use all words, or only noun and basic form of verb
all_word = False

f = open("tmp/wordmap.txt")
f.readline()

lines = [line.strip().split(" ") for line in f]
wordmap = {int(count): word for word, count in lines}

phi = np.loadtxt("tmp/model-final.phi")

filename = "topic-word%s.tsv" % ("" if all_word else "-noun")

with open(filename, "w") as f:
    for phi_t in phi:
        idx = phi_t.argsort()[::-1]
        freq = [wordmap[i] for i in idx[:30]]
        f.write("%s\n" % "\t".join(freq))
