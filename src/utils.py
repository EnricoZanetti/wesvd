#!/usr/bin/env python

# Ruff Settings
# ruff: noqa: S105, SLF001

import os

import nltk
from nltk.corpus import reuters

# Check if the operating system is Windows (os.name == 'nt')
# On some versions of Python running on Windows, SSL certificate verification
# can fail when downloading data.
if os.name == 'nt':
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

# Download the Reuters corpus using NLTK.
# This ensures that the corpus is available locally for processing.
nltk.download('reuters')

START_TOKEN = '<START>'
END_TOKEN = '<END>'

def read_corpus(category="crude"):
    """
    Read files from the specified Reuters category and processes their content.

    Each file's content is tokenized into lowercase words, with special tokens
    `<START>` and `<END>` added at the beginning and end of the word list.

    Parameters
    ----------
    category : str, optional
        The Reuters category to read from. Default is "crude".

    Returns
    -------
    list of list of str
        A list of lists, where each inner list represents the tokenized words
        from a single file in the specified category, including the `<START>`
        and `<END>` tokens.
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]
