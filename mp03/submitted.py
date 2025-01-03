'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''
    distances = []
    for i in range(len(train_images)):
        diff = image - train_images[i]
        diff = diff @ diff
        distances.append((diff, i))
    distances.sort(key=lambda x : x[0])
    distances = distances[:k]
    neighbors = np.array([train_images[i] for _, i in distances])
    labels = np.array([train_labels[i] for _, i in distances])
    return neighbors, labels

def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses = []
    scores = []
    for image in dev_images:
        neighbors, labels = k_nearest_neighbors(image, train_images, train_labels, k)
        hypotheses.append(np.count_nonzero(labels == True) > np.count_nonzero(labels == False))
        scores.append(np.count_nonzero(labels == hypotheses[-1]))
    return hypotheses, scores

def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    confusions = [[0, 0], [0, 0]]
    for i in range(len(hypotheses)):
        confusions[references[i]][hypotheses[i]] += 1
    accuracy = (confusions[0][0] + confusions[1][1]) / (confusions[0][0] + confusions[0][1] + confusions[1][0] + confusions[1][1])
    precision = confusions[1][1] / (confusions[1][1] + confusions[0][1])
    recall = confusions[1][1] / (confusions[1][1] + confusions[1][0])
    f1 = 2 / (1 / recall + 1 / precision)
    confusions = np.array(confusions)
    return confusions, accuracy, f1
