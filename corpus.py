# -*- coding: utf-8; -*-
import re

from pymongo import MongoClient
import MeCab

# use all words, or only noun and basic form of verb
all_word = False

client = MongoClient('localhost')
db = client.tweets
tagger = MeCab.Tagger()

regex = re.compile("https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")

filename = "tmp/corpus%s.tsv" % ("" if all_word else "-noun")

with open(filename, "w") as f:
    result = db.tweets.find()
    f.write("%d\n" % result.count())
    for tweet in result:
        text = tweet["text"]
        urls = tweet["entities"]["urls"]
        domains = [url["expanded_url"].split("/")[2] for url in urls]
        text = regex.sub("", text)

        hashtags = []
        for hashtag in tweet["entities"]["hashtags"]:
            hashtag = "#" + hashtag["text"]
            text = text.replace(hashtag, " ")
            hashtags.append(hashtag.encode("utf-8"))

        words = []
        for part in tagger.parse(text.encode("utf-8")).split("\n"):
            if part == 'EOS':
                break
            if all_word:
                words.append(part.split("\t")[0])
            else:
                word, feature = part.split("\t")
                features = feature.split(",")
                if features[0] == "名詞":
                    words.append(word)
                elif features[0] == "動詞":
                    orig = features[6]
                    if orig != "*":
                        words.append(orig)

        words.extend(domains)
        words.extend(hashtags)
        words = [word.decode("utf-8") for word in words]
        f.write((u"%s\n" % "\t".join(words)).encode("utf-8"))
