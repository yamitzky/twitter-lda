import csv

import numpy as np

f = open("tmp/wordmap.txt")
f.readline()

lines = [line.strip().split(" ") for line in f]
wordmap = {int(count): word for word, count in lines}

phi = np.loadtxt("tmp/model-final.phi")

with open("topic-word.txt", "w") as f:
    for phi_t in phi:
        idx = phi_t.argsort()[::-1]
        freq = [wordmap[i] for i in idx[:30]]
        f.write("%s\n" % "\t".join(freq))
