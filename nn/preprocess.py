# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate sequences by class (positive/negative labels)
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    # Find max class size
    max_class_size = max(len(pos_seqs), len(neg_seqs))

    # Upsample the minority class with replacement, so we have the same number of positive and negative examples
    if len(pos_seqs) < max_class_size:
        pos_seqs += random.choices(pos_seqs, k=max_class_size - len(pos_seqs))
    elif len(neg_seqs) < max_class_size:
        neg_seqs += random.choices(neg_seqs, k=max_class_size - len(neg_seqs))

    # Ensure all sequences are the same length by padding with "P"
    max_seq_length = max(len(seq) for seq in pos_seqs + neg_seqs)
    pos_seqs = [seq.ljust(max_seq_length, "P") for seq in pos_seqs]
    neg_seqs = [seq.ljust(max_seq_length, "P") for seq in neg_seqs]

    # Merge balanced dataset
    balanced_seqs = pos_seqs + neg_seqs
    balanced_labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    # Shuffle dataset
    combined = list(zip(balanced_seqs, balanced_labels))
    random.shuffle(combined)
    balanced_seqs, balanced_labels = zip(*combined)

    return list(balanced_seqs), list(balanced_labels)


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encoding_dict = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
        "P": [2, 2, 2, 2]  # Special padding encoding
    }

    # Encode each sequence
    encoded_sequences = []
    for seq in seq_arr:
        encoded_seq = []
        for nucleotide in seq:
            encoded_seq.extend(encoding_dict.get(nucleotide, [0, 0, 0, 0]))  # Default to zero for unknown bases
        encoded_sequences.append(encoded_seq)

    return np.array(encoded_sequences, dtype=np.float32)