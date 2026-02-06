"""Human-readable run ID generation."""

from __future__ import annotations

import random

_ADJECTIVES = [
    "brave", "calm", "clever", "cool", "crisp", "daring", "eager", "fast",
    "fierce", "flying", "funny", "gentle", "golden", "grand", "happy",
    "honest", "icy", "jolly", "keen", "kind", "lively", "lucky", "merry",
    "mighty", "noble", "orange", "pale", "pink", "plain", "proud", "quick",
    "quiet", "rapid", "red", "rich", "round", "sharp", "shiny", "silent",
    "silver", "sleek", "slim", "smooth", "snowy", "soft", "solid", "steady",
    "steep", "still", "strong", "sunny", "sweet", "swift", "tall", "tame",
    "thick", "tiny", "warm", "wild", "wise", "witty",
]

_NOUNS = [
    "badger", "bear", "bison", "cobra", "crane", "crow", "deer", "drake",
    "eagle", "falcon", "finch", "fox", "frog", "goose", "hawk", "heron",
    "horse", "husky", "ibis", "igloo", "jackal", "jaguar", "koala", "lark",
    "lemur", "lion", "llama", "lynx", "mango", "moose", "newt", "nomad",
    "otter", "owl", "panda", "pearl", "pebble", "pine", "plum", "puma",
    "quail", "raven", "reef", "robin", "sage", "seal", "shark", "snail",
    "spark", "squid", "stork", "swallow", "tiger", "trout", "tulip",
    "viper", "whale", "wolf", "wren", "yak",
]


def generate_run_id(rng: random.Random | None = None) -> str:
    """Generate a human-readable run ID.

    Pattern: ``adjective-adjective-noun`` (e.g. ``funny-flying-swallow``).

    Parameters
    ----------
    rng : random.Random | None
        Optional random number generator instance. When *None*, a fresh
        default-seeded instance is used.

    Returns
    -------
    str
        A run ID like ``"brave-calm-otter"``.
    """
    if rng is None:
        rng = random.Random()
    adj1, adj2 = rng.sample(_ADJECTIVES, 2)
    noun = rng.choice(_NOUNS)
    return f"{adj1}-{adj2}-{noun}"
