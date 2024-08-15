import argparse
from multiprocessing import Pool
from pathlib import Path
import re
import sqlite3
from typing import Callable, Iterable

from datasets import load_dataset
from doctr.datasets import VOCABS
from tqdm import tqdm
from doctr.datasets import VOCABS
from tqdm import tqdm
from lib.config import Config
from lib.constants import HANGUL_SYLLABLES, KOREAN_ALPHABET

# @todo: pre-build cumdist


def run(args):
    config = Config.load_toml(args.config_file)

    raw_words = [
        *load_ds(
            "csqa",
            dict(path="ozgur-celik/csqa_korean"),
            lambda xs: dict(text=[x for x in xs["question"]]),
            len_estimate=2_500,
        ),
        *load_ds(
            "sentiment",
            dict(path="sepidmnorozy/Korean_sentiment"),
            lambda xs: dict(text=[x for x in xs["text"]]),
            len_estimate=36_000,
        ),
        *load_ds(
            "open_subtitles",
            dict(
                path="Helsinki-NLP/open_subtitles",
                lang1="en",
                lang2="ko",
            ),
            lambda xs: dict(text=[x["ko"] for x in xs["translation"]]),
            len_estimate=1_391_190,
        ),
    ]

    vocab = dict()
    with Pool(args.workers) as p:
        word_iter = tqdm(
            p.imap_unordered(process_word, raw_words),
            desc="Counting words...",
            total=len(raw_words),
        )
        for w in word_iter:
            if w is None:
                continue

            vocab.setdefault(w, 0)
            vocab[w] += 1

    if args.latin_weight:
        for char in VOCABS["digits"] + VOCABS["ascii_letters"]:
            vocab[char] = max(vocab.get(char, 0), args.latin_weight)
    if args.hangul_weight:
        for char in HANGUL_SYLLABLES:
            vocab[char] = max(vocab.get(char, 0), args.hangul_weight)

    print(
        f"Generated vocab with {len(vocab):,} words occurring {sum(vocab.values()):,} times"
    )

    insert_vocab(config, vocab)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=Path,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--hangul-weight",
        type=int,
        default=10,
        help="Minimum frequency for each hangul syllable",
    )
    parser.add_argument(
        "--latin-weight",
        type=int,
        default=1000,
        help="Minimum frequency for each latin character (abc...ABC...XYZ...0123...)",
    )

    return parser.parse_args()


def insert_vocab(config: Config, vocab: dict):
    db = sqlite3.connect(config.vocab_file)

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS vocab (
            id      TEXT        PRIMARY KEY,
            count   INTEGER     NOT NULL
        )
        """
    )

    db.execute("DELETE FROM vocab")

    print(f"Dumping to {config.vocab_file}...")
    vocab_iter = tqdm(list(vocab.items()))
    for k, v in vocab_iter:
        db.execute(
            """
            INSERT INTO vocab (
                id, count
            ) VALUES (
                ?, ?
            )
            """,
            [k, v],
        )

    db.commit()


def load_ds(
    name: str,
    ds_kwargs: dict,
    getter: Callable[[dict], dict[str, list[str]]],
    len_estimate: int | None = None,
) -> Iterable[str]:
    ds = load_dataset(split="train", streaming=True, **ds_kwargs)

    ds_iter = tqdm(
        ds.map(getter, batched=True),
        desc=f"Loading {name} dataset...",
        total=len_estimate,
    )
    for item in ds_iter:
        for word in item["text"].split():  # type: ignore
            yield word


def process_word(word: str) -> str | None:
    w = clean(word)
    if is_valid(w):
        return w


def clean(chars: str):
    return re.sub(re.escape(VOCABS["punctuation"]), "", chars)


def is_valid(chars: str):
    return len(chars) > 0 and all(c in KOREAN_ALPHABET for c in chars)


if __name__ == "__main__":
    args = parse_args()
    run(args)
