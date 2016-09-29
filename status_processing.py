#!/bin/env python

"""train a word2vec model on text from a designated column of a CSV"""

# Python stdlib imports
import argparse
import csv
import itertools
import logging
import string

# third-party library imports
import gensim
import nltk.data
import IPython
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


logging.basicConfig(level=logging.INFO)


# XXX TODO: don't load this in global scope (`--help` is weirdly slow)
sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def to_sentences(text):
    lematizer = WordNetLemmatizer()
    sentence_strings = sentence_detector.tokenize(text)
    sentence_lists = [wordpunct_tokenize(sentence) for sentence in sentence_strings]
    for sentence in sentence_lists:
        yield [lematizer.lemmatize(word.lower()) for word in sentence
               if not all(char in string.punctuation for char in word)]


class TaggedSentenceIterable:
    def __init__(self, csv_path, tag_fieldname, text_fieldname,
                 sentencizer, limit=None):
        # path to our CSV
        self.csv_path = csv_path
        # name of column within CSV that contains tag (e.g., author) for text
        self.tag_fieldname = tag_fieldname
        # name of column within CSV that contains our text
        self.text_fieldname = text_fieldname
        # callable that yields sentences given text
        self.sentencizer = sentencizer
        # maximum number of rows to process (useful for testing less than the
        # full dataset)
        self.limit = limit

    def __iter__(self):
        count = 0
        with open(self.csv_path) as our_csv:
            reader = csv.reader(our_csv)
            header = next(reader)
            tag_field_index = header.index(self.tag_fieldname)
            text_field_index = header.index(self.text_fieldname)

            for row in reader:
                tag = row[tag_field_index]
                text = row[text_field_index]
                yield from zip(itertools.repeat(tag), self.sentencizer(text))
                count += 1

                if count % 5000 == 0:
                    logging.info("extracted sentences from %s rows so far", count)

                if self.limit is not None:
                    if count >= self.limit:
                        break


class SentenceIterable(TaggedSentenceIterable):
    """Given the path to a CSV and the name of the CSV column that
    contains text, iterate over sentences as suitable for input into
    the gensim Word2Vec model."""
    def __init__(self, csv_path, text_fieldname,
                 sentencizer, limit=None):
        super().__init__(csv_path,
                         # "tag" is ignored here, so just reuse text fieldname
                         text_fieldname,
                         text_fieldname,
                         sentencizer, limit)

    def __iter__(self):
        for _tag, sentence in super().__iter__():
            yield sentence


def build_model(csv_path, text_fieldname, limit=None, **kwargs):
    return gensim.models.Word2Vec(
        SentenceIterable(csv_path, text_fieldname, to_sentences, limit=limit),
        **kwargs
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("n", type=int, nargs='?', default=20000,
                            help=("number of rows to process "
                                  "(entire dataset==25407762)"))
    args = arg_parser.parse_args()
    model = build_model("status_updates.csv", "message",
                        limit=args.n, min_count=150)
    print("model is available in variable 'model'")
    # drop into an IPython shell for exploration
    IPython.embed()
