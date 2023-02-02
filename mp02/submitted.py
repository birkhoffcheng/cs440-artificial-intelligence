'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter
import math

stopwords = {"a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"}

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists)
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters)
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    frequency = {}
    for cla in train:
        ctr = Counter()
        for text in train[cla]:
            ctr.update(text)
        frequency[cla] = ctr
    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters)
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    for cla in frequency:
        for w in stopwords:
            if frequency[cla].get(w, None) is not None:
                del frequency[cla][w]
    return frequency

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts)
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    likelihood = {}
    for cla in nonstop:
        likelihood[cla] = {}
        total = Counter(nonstop[cla]).total()
        for w in nonstop[cla]:
            likelihood[cla][w] = (nonstop[cla][w] + smoothness) / (total + smoothness * (len(nonstop[cla]) + 1))
        likelihood[cla]['OOV'] = smoothness / (total + smoothness * (len(nonstop[cla]) + 1))
    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts)
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    for text in texts:
        max_class = None
        max_prob = None
        for cla in likelihood:
            prob = 0
            oov_likelihood = likelihood[cla]['OOV']
            if cla == 'pos':
                prob += math.log(prior)
            else:
                prob += math.log(1 - prior)
            for w in text:
                if w in stopwords:
                    continue
                prob += math.log(likelihood[cla].get(w, oov_likelihood))
            if max_prob is None or prob > max_prob:
                max_prob = prob
                max_class = cla
        hypotheses.append(max_class)
    return hypotheses

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for j in range(len(smoothnesses)):
        likelihood = laplace_smoothing(nonstop, smoothnesses[j])
        for i in range(len(priors)):
            hypotheses = naive_bayes(texts, likelihood, priors[i])
            for k in range(len(labels)):
                if hypotheses[k] == labels[k]:
                    accuracies[i, j] += 1
            accuracies /= len(labels)
    return accuracies
