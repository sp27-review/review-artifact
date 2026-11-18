"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from dataclasses import dataclass


WORDS_PATH = "/usr/share/dict/words"


@dataclass(frozen=True)
class QualityGate:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

    oov_max: float = 0.08
    allcaps_upper_frac: float = 0.9
    allcaps_min_lower: int = 40
    allcaps_singleton_oov_max: float = 0.13
    allcaps_sym_max: float = 5.0
    narrow_frac_max: float = 0.8
    narrow_width_ratio: float = 0.55
    cluster_gap_ratio: float = 0.12


@dataclass(frozen=True)
class LabelExtraction:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

    top_ratio: float = 0.15
    bottom_ratio: float = 0.85


@dataclass(frozen=True)
class Masking:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

    banner_max_words: int = 8

    fuzzy_class_forms: tuple[tuple[str, int], ...] = (
        ("TOPSECRET", 2),
        ("SECRET", 2),
        ("CONFIDENTIAL", 3),
        ("UNCLASSIFIED", 3),
    )


QUALITY_GATE = QualityGate()
LABELS = LabelExtraction()
MASKING = Masking()
