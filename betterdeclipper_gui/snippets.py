"""Where to look: the most restored peaks of a result, spread over the file.

A result is summarized per 10 ms block and channel by its largest restoration |x - y| (and where it
is), so snippets can be ranked again later without re-reading the audio. Snippets are picked greedily
by that amount; each one blocks out its surroundings (the 3 s view plus a 2 s gap on both sides), so
the list covers different moments instead of neighbouring beats of the same passage.
"""
from dataclasses import dataclass

import numpy as np

VIEW_S = 3.0        # length of the snippet view
GAP_S = 2.0         # minimum distance between two views
MAX_SNIPPETS = 10
BLOCK_S = 0.01


@dataclass(frozen=True)
class Snippet:
    center: int         # sample index of the restored peak
    channel: int
    before: float       # |input| at the peak
    after: float        # |result| at the peak
    fallback: bool = False  # nothing was restored: the input's own peak


class RestorationMap:
    """Per block and channel: the largest |x - y|, its sample index, and |y|, |x| there."""

    def __init__(self, y, x, sr, chunk=1 << 20):
        T, C = y.shape
        self.block = B = max(1, int(round(BLOCK_S * sr)))
        nb = -(-T // B)
        self.amount = np.zeros((nb, C), np.float32)
        self.pos = np.zeros((nb, C), np.int64)
        self.before = np.zeros((nb, C), np.float32)
        self.after = np.zeros((nb, C), np.float32)
        chunk = max(B, chunk // B * B)
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            ba, bb = a // B, -(-b // B)
            n = (bb - ba) * B
            for c in range(C):
                d = np.zeros(n, np.float32)
                d[:b - a] = np.abs(x[a:b, c] - y[a:b, c])
                blk = d.reshape(-1, B)
                i = blk.argmax(1)
                p = np.minimum(a + np.arange(bb - ba) * B + i, T - 1)
                self.amount[ba:bb, c] = blk[np.arange(len(i)), i]
                self.pos[ba:bb, c] = p
                self.before[ba:bb, c] = np.abs(y[p, c])
                self.after[ba:bb, c] = np.abs(x[p, c])

    @property
    def restored(self):
        return bool(self.amount.max() > 0)


def rank(rmap, sr, view_s=VIEW_S, gap_s=GAP_S, max_n=MAX_SNIPPETS):
    """Up to max_n snippets, most restored first, with centers at least view_s + gap_s apart."""
    best = rmap.amount.max(1).astype(np.float64)
    sep = int((view_s + gap_s) * sr)
    B = rmap.block
    out = []
    while len(out) < max_n:
        b = int(best.argmax())
        if best[b] <= 0:
            break
        c = int(rmap.amount[b].argmax())
        p = int(rmap.pos[b, c])
        out.append(Snippet(p, c, float(rmap.before[b, c]), float(rmap.after[b, c])))
        best[max(0, (p - sep) // B):(p + sep) // B + 1] = -1.0
    return out
