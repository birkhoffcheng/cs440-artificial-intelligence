'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    ntexts = len(texts)
    count0 = [0] * ntexts
    count1 = [0] * ntexts
    for i in range(ntexts):
        count0[i] = texts[i].count(word0)
        count1[i] = texts[i].count(word1)
    Pjoint = np.zeros((max(count0) + 1, max(count1) + 1))
    for i in range(ntexts):
        Pjoint[count0[i], count1[i]] += 1
    Pjoint /= ntexts
    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other)

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    index ^= 1
    return np.sum(Pjoint, axis=index)

def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs:
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    Pcond = np.zeros(Pjoint.shape)
    for i in range(Pjoint.shape[0]):
        for j in range(Pjoint.shape[1]):
            if Pmarginal[i] == 0:
                Pcond[i, j] = np.nan
            else:
                Pcond[i, j] = Pjoint[i, j] / Pmarginal[i]
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    mu (float) - the mean of X
    '''
    mu = 0
    for i, p in enumerate(P):
        mu += i * p
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    var (float) - the variance of X
    '''
    mu = mean_from_distribution(P)
    var = 0
    for i, p in enumerate(P):
        var += (i - mu) ** 2 * p
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)

    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    mux = mean_from_distribution(marginal_distribution_of_word_counts(P, 0))
    muy = mean_from_distribution(marginal_distribution_of_word_counts(P, 1))
    covar = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            covar += (i - mux) * (j - muy) * P[i, j]
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            expected += f(i, j) * P[i, j]
    return expected
