'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not
use other non-standard modules (including nltk). Some modules that might be helpful are
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(train, test):
	'''
	Implementation for the baseline tagger.
	input:  training data (list of sentences, with tags on the words)
			test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
	output: list of sentences, each sentence is a list of (word,tag) pairs.
			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''
	word_tags = defaultdict(Counter)
	tags_count = Counter()
	for sentence in train:
		for word, tag in sentence:
			word_tags[word][tag] += 1
			tags_count[tag] += 1
	max_tag = tags_count.most_common(1)[0][0]
	predictions = []
	for sentence in test:
		pred_sentence = []
		for word in sentence:
			if word in word_tags:
				pred_word = (word, word_tags[word].most_common(1)[0][0])
			else:
				pred_word = (word, max_tag)
			pred_sentence.append(pred_word)
		predictions.append(pred_sentence)
	return predictions


def viterbi(train, test):
	'''
	Implementation for the viterbi tagger.
	input:  training data (list of sentences, with tags on the words)
			test data (list of sentences, no tags on the words)
	output: list of sentences with tags on the words
			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''
	raise NotImplementedError("You need to write this part!")


def viterbi_ec(train, test):
	'''
	Implementation for the improved viterbi tagger.
	input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
			test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
	output: list of sentences, each sentence is a list of (word,tag) pairs.
			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''
	raise NotImplementedError("You need to write this part!")
