'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not
use other non-standard modules (including nltk). Some modules that might be helpful are
already imported for you.
'''

from collections import defaultdict, Counter
from math import log

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
	tags_words = defaultdict(Counter)
	tag_pairs = defaultdict(Counter)
	tags_count = Counter()

	# record counts of tags, tag->word emissions, tag transitions
	for sentence in train:
		for t in range(1, len(sentence)):
			word, tag = sentence[t]
			_, prev_tag = sentence[t - 1]
			tags_count[tag] += 1
			tags_words[tag][word] += 1
			tag_pairs[prev_tag][tag] += 1

	# transition and emission probabilities
	transition_matrix = defaultdict(dict)
	emission_prob = defaultdict(dict)
	transition_smoothness = 0.00001
	emission_smoothness = 0.00001
	num_tags = len(tags_count)

	for tag in tag_pairs:
		for next_tag in tags_count:
			transition_matrix[tag][next_tag] = log((tag_pairs[tag][next_tag] + transition_smoothness) / (tag_pairs[tag].total() + transition_smoothness * (num_tags + 1)))

	for tag in tags_words:
		for word in tags_words[tag]:
			emission_prob[tag][word] = log((tags_words[tag][word] + emission_smoothness) / (tags_words[tag].total() + emission_smoothness * (num_tags + 1)))

	predictions = []
	for sentence in test:
		# Initialization
		prediction = []
		trellis = [None] * len(sentence)
		trellis[0] = {tag: (transition_matrix['START'][tag], tag) for tag in tags_count}

		# Iteration: construct trellis
		for t in range(1, len(sentence)):
			word = sentence[t]
			trellis[t] = {}
			for tag in tags_count:
				max_tag_prob = float('-inf')
				max_prev_tag = None
				emission_probability = emission_prob[tag].get(word, log(emission_smoothness / (tags_count[tag] + emission_smoothness * (num_tags + 1))))
				for prev_tag in transition_matrix:
					if prev_tag == 'START':
						continue
					prob = trellis[t - 1][prev_tag][0] + transition_matrix[prev_tag][tag] + emission_probability
					if prob > max_tag_prob:
						max_tag_prob = prob
						max_prev_tag = prev_tag
				trellis[t][tag] = (max_tag_prob, max_prev_tag)

		# Termination
		max_tag_prob = float('-inf')
		best_tag = None
		for tag in trellis[-1]:
			if trellis[-1][tag][0] > max_tag_prob:
				max_tag_prob = trellis[-1][tag][0]
				best_tag = tag

		# Backtrack
		for i in range(len(sentence) - 1, 0, -1):
			prediction.append((sentence[i], best_tag))
			best_tag = trellis[i][best_tag][1]

		prediction.append(('START', 'START'))
		prediction.reverse()
		predictions.append(prediction)

	return predictions


def viterbi_ec(train, test):
	'''
	Implementation for the improved viterbi tagger.
	input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
			test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
	output: list of sentences, each sentence is a list of (word,tag) pairs.
			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''
	return viterbi(train, test)
