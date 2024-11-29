#!/usr/bin/env python

# Ruff Settings
# ruff: noqa: PTH100, PTH118

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.utils import read_corpus

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
    all_words = [word for document in corpus for word in document]
    unique_words = sorted(set(all_words))
    return unique_words, len(unique_words)


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
    word2ind : dict
        Dictionary mapping each word to its corresponding row/column index in the matrix.
    """
    words, num_words = distinct_words(corpus)
    word2ind = {word: idx for idx, word in enumerate(words)}
    m = np.zeros((num_words, num_words), dtype=np.int32)

    for document in corpus:
        doc_length = len(document)
        for idx, word in enumerate(document):
            start = max(0, idx - window_size)
            end = min(doc_length - 1, idx + window_size)
            for i in range(start, end + 1):
                if i != idx:
                    context_word = document[i]
                    m[word2ind[word], word2ind[context_word]] += 1

    return m, word2ind


def reduce_to_k_dim(m, k=2):
    """
    Reduce the dimensionality of a co-occurrence matrix to `k` dimensions using Truncated SVD.

    Parameters
    ----------
    m : numpy.ndarray
        Co-occurrence matrix of shape (num_words, num_words), where num_words is
        the number of unique words.
    k : int, optional
        Number of dimensions to reduce to. Default is 2.

    Returns
    -------
    m_reduced : numpy.ndarray
        Reduced matrix of shape (num_words, k), containing the k-dimensional word embeddings.
    """
    svd = TruncatedSVD(n_components=k, n_iter=10, random_state=4355)
    m_reduced = svd.fit_transform(m)
    return m_reduced


def normalize_embeddings(m_reduced):
    """
    Normalize word embeddings to unit length.

    Parameters
    ----------
    m_reduced : numpy.ndarray
        Matrix containing reduced embeddings.

    Returns
    -------
    m_normalized : numpy.ndarray
        Normalized embeddings where each row has unit length.
    """
    row_norms = np.linalg.norm(m_reduced, axis=1, keepdims=True)
    return m_reduced / row_norms


def plot_embeddings(m_reduced, word2ind, words, title='embeddings.png'):
    """
    Plot 2D word embeddings with improved label placement and associations.

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
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # Scatter plot of embeddings
    for word in words:
        if word in word2ind:
            idx = word2ind[word]
            x, y = m_reduced[idx, 0], m_reduced[idx, 1]
            plt.scatter(x, y, marker='x', color='red', zorder=2)

            # Label directly next to the point
            plt.annotate(
                word,
                (x, y),
                textcoords='offset points',  # Specify an offset
                xytext=(5, 2),  # Slight offset for clarity
                ha='left',
                fontsize=9,
                zorder=3,
            )

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set axis labels and title
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Word Embeddings')

    # Adjust plot margins to prevent labels from being cropped
    plt.margins(0.1)

    # Save the plot to the specified file
    plt.savefig(title, bbox_inches='tight', dpi=300)
    print(f'Plot saved to {title}')


def main(args):
    """
    Process corpus, generating embeddings, and plotting.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.
    """
    # Read the corpus
    if args.corpus_file:
        # Implement reading from a file if provided
        with open(args.corpus_file) as f:
            corpus = [line.strip().split() for line in f]
    else:
        # Use default corpus from utils
        corpus = read_corpus()

    # Compute the co-occurrence matrix
    m_co_occurrence, word2ind = compute_co_occurrence_matrix(corpus, args.window_size)
    print(f'Computed co-occurrence matrix of shape {m_co_occurrence.shape}')

    # Reduce dimensionality
    m_reduced = reduce_to_k_dim(m_co_occurrence, args.k_dim)
    m_normalized = normalize_embeddings(m_reduced)

    # Save the reduced embeddings if specified
    if args.output_embeddings:
        np.save(args.output_embeddings, m_normalized)
        print(f'Saved embeddings to {args.output_embeddings}')

    # Plot embeddings if words are provided
    if args.words_to_visualize:
        words = args.words_to_visualize
        plot_embeddings(m_normalized, word2ind, words, args.output_plot)
    else:
        print('No words provided for visualization.')


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate word embeddings from a corpus using co-occurrence matrices and Truncated SVD.'
    )
    parser.add_argument(
        '--corpus_file',
        type=str,
        default=None,
        help='Path to the corpus file. Each line should be a tokenized document.',
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=4,
        help='Context window size for co-occurrence calculation.',
    )
    parser.add_argument('--k_dim', type=int, default=2, help='Number of dimensions to reduce to.')
    parser.add_argument(
        '--output_embeddings',
        type=str,
        default='embeddings.npy',
        help='Filename to save the reduced embeddings.',
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default='embeddings.png',
        help='Filename to save the embeddings plot.',
    )
    parser.add_argument(
        '--words_to_visualize', nargs='*', default=None, help='List of words to visualize.'
    )

    args = parser.parse_args()
    mpl.use('agg')  # Use 'agg' backend for plotting without GUI
    plt.rcParams['figure.figsize'] = [10, 5]

    main(args)
