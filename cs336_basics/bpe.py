import os
import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize_chunk(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """Split a chunk of text into pre-tokens and count their frequencies."""
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    # Split on special tokens so no merges happen across them
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(pattern, chunk)
    else:
        segments = [chunk]

    for segment in segments:
        for match in re.finditer(GPT2_PAT, segment):
            word = match.group()
            token_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_freqs[token_bytes] += 1

    return dict(word_freqs)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Step 1: Initialize vocabulary
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Step 2: Pre-tokenization
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, os.cpu_count() or 1, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_freqs = _pretokenize_chunk(chunk, special_tokens)
            for word, freq in chunk_freqs.items():
                word_freqs[word] += freq

    # Step 3: Build pair counts and inverted index (pair -> set of words containing it)
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # Step 4: Merge loop
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # Pick the most frequent pair, tie-break by lexicographically greater pair
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        # Add merged token to vocab and merges list
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        # Apply merge to every word containing best_pair
        affected_words = list(pair_to_words[best_pair])
        a, b = best_pair

        for old_word in affected_words:
            freq = word_freqs[old_word]

            # Build new word by merging (a, b) -> a+b, walking left-to-right
            new_word_list: list[bytes] = []
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and old_word[i] == a and old_word[i + 1] == b:
                    new_word_list.append(new_token)
                    i += 2
                else:
                    new_word_list.append(old_word[i])
                    i += 1
            new_word = tuple(new_word_list)

            # Accumulate pair count deltas (handles repeated pairs correctly)
            delta: dict[tuple[bytes, bytes], int] = defaultdict(int)
            for i in range(len(old_word) - 1):
                delta[(old_word[i], old_word[i + 1])] -= freq
            for i in range(len(new_word) - 1):
                delta[(new_word[i], new_word[i + 1])] += freq

            for pair, d in delta.items():
                if d != 0:
                    pair_counts[pair] += d
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)

            # Update inverted index: remove old_word from all its old pairs,
            # add new_word to all its new pairs. Must happen regardless of delta,
            # since a pair may exist in both old and new (delta==0) yet the
            # specific word reference still needs to be swapped.
            old_pair_set = {
                (old_word[i], old_word[i + 1]) for i in range(len(old_word) - 1)
            }
            new_pair_set = {
                (new_word[i], new_word[i + 1]) for i in range(len(new_word) - 1)
            }
            for pair in old_pair_set:
                if pair in pair_to_words:
                    pair_to_words[pair].discard(old_word)
                    if not pair_to_words[pair]:
                        pair_to_words.pop(pair, None)
            for pair in new_pair_set:
                pair_to_words[pair].add(new_word)

            # Replace word in word_freqs
            del word_freqs[old_word]
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

    return vocab, merges


