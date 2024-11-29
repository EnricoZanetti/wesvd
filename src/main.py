#!/usr/bin/env python

# Ruff Settings
# ruff: noqa: D103, PLR2004

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

from utils import read_corpus

sys.path.append(os.path.abspath(os.path.join('..')))

def distinct_words(corpus):
    """
    Identify and counts the distinct words in a given corpus.

    Parameters
    ----------
    corpus : list of list of str
        Corpus of documents, where each document is a list of words.

    Returns
    -------
    corpus_words : list of str
        Sorted list of distinct words across the entire corpus.
    num_corpus_words : int
        Total number of distinct words across the corpus.
    """
    corpus_words = []
    num_corpus_words = 0

    # Flatten the corpus into a single list of words using a list comprehension
    all_words = [word for document in corpus for word in document]

    # Create a set of unique words
    unique_words = set(all_words)

    # Sort the unique words
    corpus_words = sorted(unique_words)

    # Get the number of unique words
    num_corpus_words = len(corpus_words)

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """
    Compute the co-occurrence matrix for a given corpus and context window size.

    Parameters
    ----------
    corpus : list of list of str
        Corpus of documents, where each document is a list of words.
    window_size : int, optional
        Size of the context window for co-occurrence calculation. Default is 4.

    Returns
    -------
    m : numpy.ndarray
        Co-occurrence matrix of shape (num_words, num_words), where num_words is the
        number of unique words.
        Each entry m[i, j] represents the count of word `j` occurring within the
        context window of word `i`.
    word2ind : dict
        Dictionary mapping each word to its corresponding row/column index in the matrix.
    """
    words, num_words = distinct_words(corpus)
    m = None
    word2ind = {}

    # Create a mapping from words to their indeces
    word2ind = {word: idx for idx, word, in enumerate(words)}

    # Initialize the co-occurrence matrix with zeros
    m = np.zeros((num_words, num_words), dtype=np.int32)

    # Iterate over each document in the corpus
    for document in corpus:
        doc_length = len(document)
        # Iterate over each word in the document
        for idx, word in enumerate(document):
            # Define the context window boundaries
            start = max(0, idx - window_size)
            end = min(doc_length - 1, idx + window_size)
            # Iterate over each position in teh context window
            for i in range(start, end + 1):
                if i != idx: # Exclude the current word
                    context_word = document[i]
                    # Increment the count for the co-occurrence
                    m[word2ind[word], word2ind[context_word]] += 1

    return m, word2ind

def reduce_to_k_dim(m, k=2):
    """
    Reduce the dimensionality of a co-occurrence matrix to `k` dimensions using Truncated SVD.

    Parameters
    ----------
    m : numpy.ndarray
        Co-occurrence matrix of shape (num_words, num_words), where num_words is the number of unique words.
    k : int, optional
        Number of dimensions to reduce to. Default is 2.

    Returns
    -------
    m_reduced : numpy.ndarray
        Reduced matrix of shape (num_words, k), containing the k-dimensional word embeddings.
        The result is the product of the first k left singular vectors and the diagonal matrix of the singular values.
    """
    np.random.seed(4355)
    n_iter = 10     # Use this parameter in your call to `TruncatedSVD`
    m_reduced = None
    print(f'Running Truncated SVD over {m.shape[0]} words...')

    # Initialize the TruncatedSVD model
    svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=4355)

    # Fit the model to the co-occurrence matrix and transform it
    m_reduced = svd.fit_transform(m)

    print("Done.")
    return m_reduced

def main():
    matplotlib.use('agg')
    plt.rcParams['figure.figsize'] = [10, 5]

    assert sys.version_info[0] == 3
    assert sys.version_info[1] >= 5

    def plot_embeddings(m_reduced, word2ind, words, title):
        """
        Plot 2D word embeddings.

        Parameters
        ----------
        m_reduced : numpy.ndarray
            2D matrix containing reduced word embeddings.
        word2ind : dict
            Dictionary mapping words to their corresponding indices in m_reduced.
        words : list of str
            Words to be plotted.
        title : str
            File name to save the plot.
        """
        for word in words:
            idx = word2ind[word]
            x = m_reduced[idx, 0]
            y = m_reduced[idx, 1]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, word, fontsize=9)
        plt.savefig(title)

    #Read in the corpus
    reuters_corpus = read_corpus()

    m_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    m_reduced_co_occurrence = reduce_to_k_dim(m_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    m_lengths = np.linalg.norm(m_reduced_co_occurrence, axis=1)
    m_normalized = m_reduced_co_occurrence / m_lengths[:, np.newaxis] # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(m_normalized, word2ind_co_occurrence, words, 'co_occurrence_embeddings_(soln).png')

if __name__ == "__main__":
    main()
