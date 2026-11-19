"""Clinical patient deduplication and identifier collision checks."""

from typing import List, Tuple
import pandas as pd


class Deduplicator:
    """Removes identical patient profiles and exact record duplicates."""

    @staticmethod
    def deduplicate(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
        initial_len = len(df)
        deduped = df.drop_duplicates(subset=subset)
        removed = initial_len - len(deduped)
        return deduped, removed
