"""Vietnamese text → syllable ids for TiHon-TTS.

Every syllable becomes 7 integers:

    [onset, medial, nucleus, coda, tone, boundary, punctuation]

The first five are phonological categories of the written syllable; `boundary` and `punctuation` describe what
follows it. Text is first rewritten into speakable words (numbers, dates, times, money, units, e-mail, acronyms),
then split into syllables. Tokens that are not Vietnamese syllables are spelled with Vietnamese letter names.
"""

from __future__ import annotations

import re
import unicodedata

# ── vocabularies (the order is fixed by the trained model) ─────────────────────────────────────────────────────────
ONSETS = ["NONE", "B", "M", "PH", "V", "T", "TH", "Đ", "N", "D", "GI", "R", "X", "S", "CH", "TR", "NH", "L", "K",
          "KH", "NG", "G", "H", "P", "Q"]
MEDIALS = ["NONE", "W"]
NUCLEI = ["A", "Ă", "Â", "E", "Ê", "I", "O", "Ô", "Ơ", "U", "Ư", "IÊ", "UÔ", "ƯƠ", "OO"]
CODAS = ["NONE", "M", "N", "NG", "NH", "P", "T", "C", "CH", "J", "W"]
TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
BOUNDARIES = ["NONE", "WORD", "PHRASE", "SENTENCE"]
PUNCTUATION = ["NONE", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "OTHER"]
VOCAB_SIZES = tuple(len(v) for v in (ONSETS, MEDIALS, NUCLEI, CODAS, TONES, BOUNDARIES, PUNCTUATION))

# ── written syllable → categories ──────────────────────────────────────────────────────────────────────────────────
_TONE_OF_MARK = {"̀": "HUYEN", "́": "SAC", "̉": "HOI", "̃": "NGA", "̣": "NANG"}
# written onsets, longest first; "gi" and "qu" are resolved in `_split_onset`
_ONSET_OF = {"ngh": "NG", "ng": "NG", "gh": "G", "kh": "KH", "nh": "NH", "ph": "PH", "th": "TH", "tr": "TR",
             "ch": "CH", "b": "B", "c": "K", "k": "K", "d": "D", "đ": "Đ", "g": "G", "h": "H", "l": "L", "m": "M",
             "n": "N", "p": "P", "r": "R", "s": "S", "t": "T", "v": "V", "x": "X"}
_ONSET_KEYS = sorted(_ONSET_OF, key=len, reverse=True)
_VOWELS = set("aăâeêioôơuưy")
_NUCLEUS_OF = {"a": "A", "ă": "Ă", "â": "Â", "e": "E", "ê": "Ê", "i": "I", "y": "I", "o": "O", "ô": "Ô", "ơ": "Ơ",
               "u": "U", "ư": "Ư", "oo": "OO", "iê": "IÊ", "yê": "IÊ", "ia": "IÊ", "ya": "IÊ", "uô": "UÔ", "ua": "UÔ",
               "ươ": "ƯƠ", "ưa": "ƯƠ"}
# vowel groups that begin with the labial glide /w/ (written o or u): group → nucleus spelling
_GLIDE_GROUPS = {"oa": "a", "oă": "ă", "oe": "e", "uâ": "â", "uê": "ê", "uy": "y", "uơ": "ơ", "uyê": "yê",
                 "uya": "ya"}
_CONSONANT_CODAS = ("ng", "nh", "ch", "c", "m", "n", "p", "t")
_CODA_OF = {"ng": "NG", "nh": "NH", "ch": "CH", "c": "C", "m": "M", "n": "N", "p": "P", "t": "T", "i": "J", "y": "J",
            "o": "W", "u": "W"}
_W_NUCLEI = {"A", "Ă", "Â", "E", "Ê", "I", "Ơ", "IÊ"}  # /w/ never precedes a rounded nucleus
_STOP_CODAS = {"P", "T", "C", "CH"}


def _vowel_group(v: str, w_onset: bool) -> tuple[str, str] | None:
    """Vowel letters between onset and coda → (medial, nucleus) categories, or None."""
    if w_onset:  # after "qu" the u already is the glide
        return ("W", _NUCLEUS_OF[v]) if v in _NUCLEUS_OF else None
    if v in _GLIDE_GROUPS:
        return "W", _NUCLEUS_OF[_GLIDE_GROUPS[v]]
    if v in _NUCLEUS_OF:
        return "NONE", _NUCLEUS_OF[v]
    return None


def _split_onset(s: str) -> tuple[str, str, bool]:
    """→ (onset category, rest of the syllable, whether the onset already contains the /w/ glide)."""
    if s.startswith("gi"):
        # "gi" + vowel (già, giữ): gi is the onset. Before ê, a consonant or at the end (giếng, gìn, gì) the i is the
        # vowel, but the onset is still /z/ (GI).
        nxt = s[2:3]
        return ("GI", s[2:], False) if nxt in _VOWELS and nxt != "ê" else ("GI", s[1:], False)
    if s.startswith("qu"):
        # quốc: the u belongs to the diphthong uô; elsewhere (qua, quy, quên) it is the glide /w/
        return ("Q", s[1:], False) if s[2:3] == "ô" else ("Q", s[2:], True)
    for key in _ONSET_KEYS:
        if s.startswith(key):
            return _ONSET_OF[key], s[len(key):], False
    return "NONE", s, False


def syllable_categories(word: str) -> tuple[str, str, str, str, str] | None:
    """Written Vietnamese syllable → (onset, medial, nucleus, coda, tone), or None if it is not one."""
    tone, letters = "NGANG", []
    for ch in unicodedata.normalize("NFD", word.lower()):
        if ch in _TONE_OF_MARK:
            if tone != "NGANG" and tone != _TONE_OF_MARK[ch]:
                return None
            tone = _TONE_OF_MARK[ch]
        else:
            letters.append(ch)
    s = unicodedata.normalize("NFC", "".join(letters))
    if not s or not all(c in _VOWELS or c in "bcdđghklmnpqrstvx" for c in s):
        return None
    onset, rest, w_onset = _split_onset(s)
    coda, vowels = "NONE", rest
    for c in _CONSONANT_CODAS:
        if rest.endswith(c) and len(rest) > len(c):
            coda, vowels = _CODA_OF[c], rest[: -len(c)]
            break
    group = _vowel_group(vowels, w_onset)
    if group is None and coda == "NONE" and len(vowels) >= 2 and vowels[-1] in "iyou":
        # final i/y/o/u is a glide coda when the remaining letters form a vowel group (ai, ao, oai, ươu ...)
        head = vowels[:-1]
        if _vowel_group(head, w_onset) is not None:
            group, coda = _vowel_group(head, w_onset), _CODA_OF[vowels[-1]]
            if group[1] == "A" and vowels[-1] in "yu":
                group = (group[0], "Ă")  # "ay", "au", "oay" are written with a but have the short vowel ă
    if group is None:
        return None
    medial, nucleus = group
    if (medial == "W" and nucleus not in _W_NUCLEI) or (coda in _STOP_CODAS and tone not in ("SAC", "NANG")):
        return None
    return onset, medial, nucleus, coda, tone


# ── text normalisation ─────────────────────────────────────────────────────────────────────────────────────────────
DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _two_digits(n: int) -> str:
    tens, unit = divmod(n, 10)
    if tens == 0:
        return DIGITS[unit]
    head = "mười" if tens == 1 else DIGITS[tens] + " mươi"
    if unit == 0:
        return head
    tail = {1: "mốt" if tens > 1 else "một", 4: "tư" if tens > 1 else "bốn", 5: "lăm"}.get(unit, DIGITS[unit])
    return f"{head} {tail}"


def _three_digits(n: int, inner: bool) -> str:
    """0–999; `inner` = a higher group was already read, so a missing hundreds digit is said ("không trăm")."""
    hundreds, rest = divmod(n, 100)
    if hundreds == 0 and not inner:
        return _two_digits(rest)
    words = DIGITS[hundreds] + " trăm"
    if rest == 0:
        return words
    return f"{words} linh {DIGITS[rest]}" if rest < 10 else f"{words} {_two_digits(rest)}"


def number_words(n: int) -> str:
    """Integer → Vietnamese words (Northern reading: nghìn, linh, tư)."""
    if n < 0:
        return "âm " + number_words(-n)
    if n == 0:
        return DIGITS[0]
    if n >= 10 ** 9:
        high, low = divmod(n, 10 ** 9)
        words = number_words(high) + " tỷ"
        return words + (" " + _below_billion(low, inner=True) if low else "")
    return _below_billion(n, inner=False)


def _below_billion(n: int, inner: bool) -> str:
    parts = []
    for scale, size in (("triệu", 10 ** 6), ("nghìn", 10 ** 3), ("", 1)):
        group, n = divmod(n, size)
        if group:
            parts.append((_three_digits(group, inner) + " " + scale).strip())
            inner = True
    return " ".join(parts)


def _digits_words(s: str) -> str:
    return " ".join(DIGITS[int(c)] for c in s if c.isdigit())


def _decimal_words(s: str) -> str:
    whole, frac = re.split(r"[.,]", s, maxsplit=1)
    return f"{number_words(int(whole))} phẩy {_digits_words(frac)}"


def _int_words(s: str) -> str:
    return _digits_words(s) if len(s) > 1 and s[0] == "0" else number_words(int(s))


def _num_words(s: str) -> str:
    return _decimal_words(s) if re.search(r"[.,]", s) else _int_words(s)


LETTER_NAMES = {"a": "a", "ă": "á", "â": "ớ", "b": "bê", "c": "xê", "d": "dê", "đ": "đê", "e": "e", "ê": "ê",
                "f": "ép", "g": "giê", "h": "hát", "i": "i", "j": "gi", "k": "ca", "l": "e lờ", "m": "em mờ",
                "n": "en nờ", "o": "o", "ô": "ô", "ơ": "ơ", "p": "pê", "q": "quy", "r": "e rờ", "s": "ét xì",
                "t": "tê", "u": "u", "ư": "ư", "v": "vê", "w": "vê kép", "x": "ích xì", "y": "i dài", "z": "dét"}
# A few frequent loanwords; anything else that is not Vietnamese is spelled letter by letter.
LOANWORDS = {"ok": "ô kê", "okay": "ô kê", "email": "i meo", "online": "on lai", "offline": "óp lai",
             "video": "vi đê ô", "internet": "in tơ nét", "wifi": "oai phai", "app": "áp", "website": "oép sai",
             "web": "oép", "facebook": "phây búc", "youtube": "diu túp", "zalo": "da lô", "google": "gu gồ",
             "iphone": "ai phôn", "laptop": "láp tóp", "file": "phai", "pin": "pin", "covid": "cô vít"}
UNITS = {"km": "ki lô mét", "m": "mét", "cm": "xen ti mét", "mm": "mi li mét", "kg": "ki lô gam", "g": "gam",
         "mg": "mi li gam", "l": "lít", "ml": "mi li lít", "h": "giờ", "s": "giây", "kw": "ki lô oát", "w": "oát",
         "°c": "độ xê", "km/h": "ki lô mét trên giờ", "m/s": "mét trên giây"}
_UNIT_RE = "|".join(sorted((re.escape(u) for u in UNITS), key=len, reverse=True))
_NUM = r"\d+(?:[.,]\d+)?"


def _spell(token: str) -> str:
    """Letters → Vietnamese letter names, digit runs → numbers."""
    out = []
    for run in re.findall(r"\d+|[^\W\d_]+", token):
        if run.isdigit():
            out.append(_int_words(run))
        else:
            out.extend(LETTER_NAMES.get(c, "") for c in run.lower())
    return " ".join(w for w in out if w)


def normalize(text: str) -> str:
    """Rewrite numbers, dates, times, money, percentages, units, e-mails and symbols into words."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("…", "...").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
                  lambda m: " a còng ".join(" chấm ".join(p.split(".")) for p in m.group().split("@")), text)
    text = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b",
                  lambda m: f"ngày {_int_words(m[1].lstrip('0') or '0')} tháng {_int_words(m[2].lstrip('0') or '0')}"
                            f" năm {number_words(int(m[3]))}", text)
    text = re.sub(r"\b(\d{1,2})(?::|h)(\d{2})\b", lambda m: f"{number_words(int(m[1]))} giờ "
                  f"{number_words(int(m[2]))} phút" if int(m[2]) else f"{number_words(int(m[1]))} giờ", text)
    text = re.sub(r"\b(\d{1,2})h\b", lambda m: f"{number_words(int(m[1]))} giờ", text)
    text = re.sub(r"(?:\+84|\b0)\d{9,10}\b", lambda m: _digits_words(m.group().replace("+84", "0")), text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", lambda m: m.group().replace(".", ""), text)  # 1.250.000 → 1250000
    text = re.sub(rf"({_NUM})\s*(?:đồng|đ|vnđ|vnd)\b", lambda m: _num_words(m[1]) + " đồng", text, flags=re.I)
    text = re.sub(rf"\$\s*({_NUM})|({_NUM})\s*\$", lambda m: _num_words(m[1] or m[2]) + " đô la", text)
    text = re.sub(rf"({_NUM})\s*%", lambda m: _num_words(m[1]) + " phần trăm", text)
    text = re.sub(rf"({_NUM})\s*[-–]\s*({_NUM})", lambda m: f"{_num_words(m[1])} đến {_num_words(m[2])}", text)
    text = re.sub(r"\b(\d+)/(\d+)\b", lambda m: f"{_int_words(m[1])} phần {_int_words(m[2])}", text)
    text = re.sub(rf"({_NUM})\s*({_UNIT_RE})(?![\w/])", lambda m: _num_words(m[1]) + " " + UNITS[m[2].lower()],
                  text, flags=re.I)
    text = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", " cộng ", text)
    text = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", " nhân ", text)
    text = re.sub(r"(?<=\w)\s*=\s*(?=\w)", " bằng ", text)
    text = re.sub(r"(?<=\w)\s*&\s*(?=\w)", " và ", text)
    text = re.sub(_NUM, lambda m: f" {_num_words(m.group())} ", text)  # spaces: "5G" → "năm G"
    return re.sub(r"\s+", " ", text).strip()


# ── syllables with boundaries and punctuation ──────────────────────────────────────────────────────────────────────
_TOKEN = re.compile(r"[^\W_]+|[^\w\s]")


def _words_of(token: str) -> list[tuple[str, str, str, str, str]]:
    cats = syllable_categories(token)
    if cats is not None:
        return [cats]
    reading = LOANWORDS.get(token.lower()) or _spell(token)
    out = [syllable_categories(w) for w in reading.split()]
    return [c for c in out if c is not None] or [("NONE", "NONE", "A", "NONE", "NGANG")]


def text_to_syllables(text: str) -> list[list[int]]:
    """One chunk of text → syllable ids (N, 7)."""
    rows: list[list] = []  # [categories, boundary, punctuation string]
    pending = ""
    for tok in _TOKEN.findall(normalize(text)):
        if not tok[0].isalnum():
            if not rows:
                pending += tok
                continue
            rows[-1][2] += tok
            if tok in ".!?":
                rows[-1][1] = "SENTENCE"
            elif tok in ",;:":
                rows[-1][1] = "PHRASE"
            continue
        new = [[c, "WORD", ""] for c in _words_of(tok)]
        new[0][2], pending = pending, ""
        rows.extend(new)
    ids = []
    for (onset, medial, nucleus, coda, tone), boundary, p in rows:
        punct = ("QUESTION" if "?" in p else "EXCLAMATION" if "!" in p else "COMMA" if "," in p
                 else "PERIOD" if "." in p else "OTHER" if p else "NONE")
        ids.append([ONSETS.index(onset), MEDIALS.index(medial), NUCLEI.index(nucleus), CODAS.index(coda),
                    TONES.index(tone), BOUNDARIES.index(boundary), PUNCTUATION.index(punct)])
    return ids


def split_text(text: str, max_chars: int = 400) -> list[str]:
    """Split long text into chunks at sentence, then clause, then word boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        parts = [sentence] if len(sentence) <= max_chars else re.split(r"(?<=[,;:])\s+", sentence)
        for part in parts:
            while len(part) > max_chars:
                cut = part.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(part[:cut])
                part = part[cut:].strip()
            if part:
                pieces.append(part)
    chunks, current = [], ""
    for piece in pieces:  # greedily re-join short pieces
        if current and len(current) + 1 + len(piece) <= max_chars:
            current += " " + piece
        else:
            if current:
                chunks.append(current)
            current = piece
    return chunks + ([current] if current else [])


def text_to_ids(text: str) -> list[list[list[int]]]:
    """Text → list of chunks, each a list of syllable ids (N, 7)."""
    return [ids for ids in (text_to_syllables(c) for c in split_text(text)) if ids]
