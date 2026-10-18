import math

from warraq.core import recognize as R
from warraq.core.document import BODY, Region
from warraq.core.imaging import LineBand
from warraq.core.text import clean_model_output, drop_foreign_scripts, foreign_char_count
from warraq.models.ocr_model import Recognition, line_stats
from warraq.settings import Settings

lp = math.log


def test_line_stats_split_lines_and_ignore_markup():
    stats = line_stats(["<", "table", "><", "td", ">", "بلغ", " مقابلة", "</", "td", ">"], [lp(0.05)] * 5 + [lp(0.9), lp(0.8)] + [lp(0.1)] * 3)
    assert len(stats) == 1 and abs(stats[0][0] - 0.8) < 1e-6  # markup tokens never make a line doubtful
    stats = line_stats(["ي", "ومن", " يعمل", "\n", "وقد", " كان", "\n\n", "x"], [lp(0.04), lp(0.4), lp(0.9), lp(0.99), lp(0.6), lp(0.9), lp(0.9), lp(0.3)])
    assert [round(p, 2) for p, _ in stats] == [0.04, 0.6, 0.3]


def _chunk(text, stats, bands=1, ok=True):
    region = Region(id="r", kind=BODY, label="text", bbox=(0, 0, 100, 100))
    c = R.Chunk(region, 0, [LineBand(i * 10, i * 10 + 8, 0, 100) for i in range(bands)], (0, 0, 100, 100), ok=ok)
    c.result = Recognition(text=text, confidence=math.exp(sum(m for _, m in stats) / len(stats)), tokens=10, hit_limit=False, looped=False, line_stats=stats)
    return c


def test_patches_replace_only_more_confident_lines():
    block = _chunk("السطر الأول\nيومن يعمل بما علم\n\nالسطر الثالث", [(0.9, -0.02), (0.04, -0.15), (0.8, -0.03)], bands=3)
    good = _chunk("وأن يعمل بما علم", [(0.3, -0.05)])
    worse = _chunk("السطر", [(0.5, -0.4)])
    text, changed = R._apply_patches(block, {1: good, 2: worse})
    assert changed
    assert text == "السطر الأول\nوأن يعمل بما علم\n\nالسطر الثالث"  # blank line kept, line 3 untouched
    # a re-read that lost half the line is never taken, however confident
    short = _chunk("وأن", [(0.99, -0.001)])
    assert R._apply_patches(block, {1: short}) == (block.result.text, {})


def test_weak_words_by_line():
    from warraq.models.ocr_model import weak_words

    pieces = ["<", "td", ">", "ي", "ومن", " يعمل", "\n", "وقد", " كان", "</", "td", ">"]
    probs = [0.01, 0.01, 0.01, 0.04, 0.4, 0.9, 0.99, 0.95, 0.2, 0.01, 0.01, 0.01]
    assert weak_words(pieces, probs, 0.3) == [(0, "يومن"), (1, "كان")]


def test_review_words_match_final_text():
    assert R._review_words(["«يومن،", "كان", "كان", "غائب"], "يومن يعمل، وكان") == ["يومن", "كان"]


def test_orientation_uses_the_best_attempt():
    first = _chunk("بلغ مقابلة", [(0.9, -0.01)])
    retry = _chunk("黑白三色", [(0.2, -1.5)], ok=False)
    retry.previous = first
    job = R.RegionJob(first.region, None, (0, 0), (10, 10), -90, 30, first.bands, chunks=[retry])
    assert R._best_attempt(retry) is first
    assert R._job_score(job) == R._chunk_score(first)


def test_foreign_script_cleanup():
    assert foreign_char_count("沌 يتكلمون") == 1
    assert drop_foreign_scripts("沌 يتكلمون") == ("يتكلمون", True)
    assert drop_foreign_scripts("قال：نعم，ولا？")[0] == "قال:نعم،ولا؟"
    assert drop_foreign_scripts("Ελληνικά и русский") == ("Ελληνικά и русский", False)
    text, notes = clean_model_output("沌 يتكلمون إلا بما ينفع", Settings())
    assert text == "يتكلمون إلا بما ينفع" and "removed misread characters" in notes
