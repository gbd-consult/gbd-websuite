"""Client-side navigation and search index.

This module builds the three static files that drive the generated site's table of
contents, navigation and full-text search.

Matching runs against a compact inverted index; the original text is never resident as
strings — it is reconstructed on demand, only for the handful of results actually shown,
from a stream of integer token ids that reference a shared dictionary.


## Files

Three artifacts are written to `staticDir`, all fetched lazily on the first search:

- `_toc.json`  — the section tree, for navigation and result titles.
- `_index.json` — the dictionary, punctuation table, tokenizer policy and the binary
                  block descriptors. Everything that is naturally JSON.
- `_index.bin` — the numeric index: inverted index (matching) and token streams
                  (snippet reconstruction), as little-endian typed arrays.

Section numeric ids (`nid`) are dense integers `1 .. numSections`, assigned in a stable
order so builds are reproducible. The same nids are used in `_toc.json`, in the postings and
to index the token streams. The string id (`sid`, e.g. `/db`) is the human-facing identifier
the rest of dog speaks in.


## _toc.json

    {
      "sections": [
        { "nid": 1, "sid": "/", "title": "…", "url": "/index.html",
          "parent": null, "children": [2, 7, 12] },
        …
      ]
    }

A section may carry an optional `"tocDepth": n` — how deep the sidebar shows it, counting the
section itself (1 = no children, absent = unlimited).

The client builds three maps at load: `byNid`, `byUrl` (url → section) and `bySid`
(sid → section). The tree is rendered from the root (the section with `parent == null`).


## _index.json

    {
      "version": 1,
      "counts": { "words": 35814, "sections": 1200 },

      "words": ["", "Abfrage", "abfragen", "…"],   // surface forms, see "Dictionary" (1-based)
      "punct": ["", " ", "", ", ", ". ", " (", ") ", "/", "-", …],  // punct id p -> punct[p]

      "tokenizer": { "blendChars": "-/._&", "extraChars": "<>" },  // shared policy; the client
                                                // rebuilds the word regex from these in its dialect

      "bin": {                                  // element offsets into _index.bin regions
        "word2secMap": { "off": 0,      "count": 35816, "type": "u32" },
        "word2sec":     { "off": 143264, "count": …,     "type": "u16" },
        "sec2tokenMap": { "off": …,      "count": 1202,  "type": "u32" },
        "sec2token":     { "off": …,      "count": …,     "type": "u16" }
      }
    }

Every id-indexed array carries a dummy element at index 0, so a token id, word number or
section nid indexes it directly — no `- 1` offsets. `words[0]` and `punct[0]` are `""`.

`off` is a byte offset into `_index.bin`; `count` is the number of elements. Each region is
padded so its byte offset is aligned to its element size (u32 on 4 bytes, u16 on 2), so the
client can wrap it zero-copy.


## Token ids

A token id is a single integer that names either a word or a run of punctuation:

    id == 0          reserved (sentinel / none; never appears in a stream)
    1 <= id <= 100   punctuation; the string is `punct[id]` from _index.json
    id  > 100        word; the surface form is `words[id - 100]`

Word ids therefore start at 101 and run `100 + wordNum` (word numbers `1 .. numWords`). Both
arrays are 1-based with a dummy element at index 0, so a token id decodes to text with a
direct lookup — `punct[id]` or `words[id - 100]`.


### Dictionary (`words`)

Surface forms, stored as-is — no case folding, so snippets keep `PostgreSQL`, `WebSuite`,
`Heinz-Müller-Straße` intact. Index 0 is the empty-string dummy; words `1 .. numWords` are
sorted by the *lowercased* form (`word.toLowerCase()`), which is the invariant the search
bsearch depends on. The `""` at index 0 sorts before every real word, so it never falls in a
prefix range. Case variants (`Config`, `config`) sort adjacent, so a prefix range already
unites them — no separate fold map is needed.

Matching is prefix-based: a query term matches a dictionary word when
`word.toLowerCase().startsWith(term)`. 


### Punctuation table (`punct`)

`punct[id]` is the literal string a punctuation token expands to, *including any spaces it
carries*. The table holds the spaced and unspaced variants a mark actually needs — `","` vs
`", "`, `"("` vs `" ("`, `")"` vs `") "` — because inter-word spacing that sits next to
punctuation must live inside the token (see decoding). A bare single space between two words
is NOT a token; it is implicit. Blend characters (`-`, `/`) are tight punctuation with no
surrounding space, so `buffer-size` and `Heinz-Müller-Straße` round-trip exactly.

Any glue run the indexer meets that is not in the table is replaced with a single space.


## _index.bin

Four concatenated regions, little-endian, each aligned to its element size. Offsets and
counts are in `_index.json` under `bin`; there is no binary header.

    1. word2secMap   (numWords + 2) x u32    element offsets into `word2sec`
    2. word2sec       * x u16                 flat section-nid lists (the inverted index)
    3. sec2tokenMap   (numSections + 2) x u32 element offsets into `sec2token`
    4. sec2token       * x u16                 flat token-id streams (one run per section)

Offsets are *element* offsets. Each map carries a dummy slot 0 so a 1-based word number / section
nid indexes it directly (the trailing entry closes the last range):

    word w  (w = tokenId - 100, 1..numWords) has postings   word2sec[ word2secMap[w] : word2secMap[w+1] ]
    section s (nid 1..numSections)            has token stream  sec2token[ sec2tokenMap[s] : sec2tokenMap[s+1] ]

Postings for each word are section nids sorted ascending, so AND queries are a linear merge.
No positions are stored: a match is located, when needed, by scanning the section's token
stream for the query's word ids (an integer scan, not a text search).

Widths: section ids and token ids are u16 (corpus vocabulary ~36k, well under 65,536; verify
the section count stays under 65,536 as the API reference is generated). Offsets are u32.


## Tokenizer (indexer and client must agree)

Index-time and query-time analysis must produce the same tokens, or a query silently fails
to match what was indexed. The policy is data in `_index.json.tokenizer` and is applied
identically on both sides — the one place the same rules exist in two languages (Python here,
JS in the theme), so it is kept small and pinned.

Blend characters (Sphinx's `blend_chars`) are indexed both ways: `buffer-size` yields the
parts `buffer`, `size` AND the joined `buffer-size`; `trim` additionally emits the joined form
with a leading blend run stripped, so `--buffer-size` is also findable as `buffer-size`. The
joined forms are matching-only aliases — they appear in the dictionary and postings but never
in a token stream, which always carries the fine-grained words so snippets reconstruct exactly.

Extra characters (`extra_chars`) instead widen the word-character class: they neither split a
word nor add a joined form, so `<h1>` (with `<`, `>` as extras) is a single token `<h1>`. Both
sides build their word regex as word-chars `∪ extra_chars`, split on blend runs.


## Search algorithm (client)

    1. analyze(query) -> terms
         Tokenize with the shared policy, lowercase. Emit word parts and, for blend runs,
         the joined and trimmed forms.

    2. resolve(term) -> wordIds
         Two bsearches over `words` (comparing on the lowercased form) bracket the prefix
         range [lo, hi) of 1-based word numbers: lo = first word whose lower form >= term,
         hi = first that no longer starts with term. wordIds = { 100 + w : lo <= w < hi }.
         ~log2(numWords).

    3. postings(term) = union over its wordIds of postings[...] slices (section-nid sets).

    4. combine
         Intersect the per-term section sets (all terms present). If empty, fall back to the
         union (any term). Rank joined/exact-term hits above part-only hits. Keep the top
         `maxResults` sections.

    5. snippet(section, matchedWordNums)   for each shown result
         Read the section's token stream. Find the first index whose word number (tokenId - 100)
         is in matchedWordNums. Take a window of tokens around it, decode to text, wrap the matched
         word tokens in <mark>, and add leading/trailing ellipses when the window is clipped.


## Decoding a token stream to text

    lastWasWord = false
    for id in window:
        if id <= 100:
            emit punct[id]          # carries its own spaces
            lastWasWord = false
        else:
            if lastWasWord:
                emit ' '            # the implicit single space between two words
            emit words[id - 100]
            lastWasWord = true

Word tokens whose id is in the matched set are wrapped in <mark> as they are emitted; the
surrounding text is HTML-escaped first.


## Navigation (client)

From `_toc.json`: render the tree from the root; mark the active section by matching
`location.pathname + location.hash` against `url` (via `byUrl`); open its ancestor chain via
`parent`. The prev/next arrows page through the files in natural reading order (`prev`/`next`
walk the file-owning sections in document order, so they skip sections sharing the active file);
the up arrow is `parent`. `bySid` resolves the `:sid` cross-references that the rest of dog
speaks in.
"""

import os
import re
import json
import struct

from . import util as u
from .types import BaseBuilder, Section, MarkdownNode, Resource


VERSION = 1
PUNCT_MAX = 100


def _patterns(blend_chars: str, extra_chars: str) -> tuple[str, str]:
    wc = rf'(?:[^\W_]|[{re.escape(extra_chars)}])' if extra_chars else r'[^\W_]'
    if not blend_chars:
        return rf'{wc}+', r'(?!)'
    cc = re.escape(blend_chars)
    word_pat = rf'{wc}+(?:[{cc}]+{wc}+)*'
    blend_pat = rf'([{cc}]+)'
    return word_pat, blend_pat


## public

def make_toc(b: BaseBuilder, save: bool = False) -> str:
    secs = _ordered_sections(b)
    nid_of = {sec.sid: i + 1 for i, sec in enumerate(secs)}
    #< options are guaranteed to have all keys, remove this var
    toc_depth = b.options.tocDepth or {}

    sections = []
    for i, sec in enumerate(secs):
        #< just write 'tocDepth': b.options.tocDepth.get(sec.sid), remove "entry" 
        entry = {
            'nid': i + 1,
            'sid': sec.sid,
            'title': sec.headText,
            'url': sec.htmlUrl,
            'parent': nid_of.get(sec.parentSid) if sec.parentSid else None,
            'children': [nid_of[s] for s in (sec.subSids or []) if s in nid_of],
        }
        if sec.sid in toc_depth:
            entry['tocDepth'] = toc_depth[sec.sid]
        sections.append(entry)

    js = _dump({'sections': sections})
    if save:
        u.write_file(_static_path(b, Resource.TOC_JSON), js)
    return js


def make_index(b: BaseBuilder, save: bool = False) -> tuple[str, bytes]:
    secs = _ordered_sections(b)

    blend_chars = b.options.blendChars
    extra_chars = b.options.extraChars
    word_pat, blend_pat = _patterns(blend_chars, extra_chars)

    sec_items = []
    sec_matches = []
    glue_freq = {}

    for sec in secs:
        items, matches = _tokenize(_section_text(sec), word_pat, blend_pat)
        sec_items.append(items)
        sec_matches.append(matches)
        for kind, val in items:
            if kind == 'g' and val.strip():
                glue_freq[val] = glue_freq.get(val, 0) + 1

    glues = sorted(glue_freq, key=lambda g: (-glue_freq[g], g))[:PUNCT_MAX]
    punct_id = {g: i + 1 for i, g in enumerate(glues)}
    punct_arr = [''] + list(glues)

    surfaces = sorted(set().union(*sec_matches) if sec_matches else set(), key=lambda s: (s.lower(), s))
    word_id = {s: PUNCT_MAX + 1 + i for i, s in enumerate(surfaces)}
    words_arr = [''] + list(surfaces)

    n_words = len(surfaces)
    n_secs = len(secs)

    if n_secs > 0xffff or PUNCT_MAX + n_words >= 0xffff:
        u.log.error('search index: id space exceeds u16, index will be corrupt')

    word_secs = {s: [] for s in surfaces}
    for i, matches in enumerate(sec_matches):
        nid = i + 1
        for s in matches:
            word_secs[s].append(nid)

    streams = []
    for items in sec_items:
        ids = []
        for kind, val in items:
            if kind == 'w':
                ids.append(word_id[val])
            else:
                pid = punct_id.get(val)
                if pid is not None:
                    ids.append(pid)
        streams.append(ids)

    word2sec = []
    word2secMap = [0] * (n_words + 2)
    for i, surface in enumerate(surfaces):
        word2secMap[i + 1] = len(word2sec)
        word2sec.extend(sorted(word_secs[surface]))
    word2secMap[n_words + 1] = len(word2sec)

    sec2token = []
    sec2tokenMap = [0] * (n_secs + 2)
    for i, stream in enumerate(streams):
        sec2tokenMap[i + 1] = len(sec2token)
        sec2token.extend(stream)
    sec2tokenMap[n_secs + 1] = len(sec2token)

    blob, bin_desc = _pack_regions([
        ('word2secMap', word2secMap, 4),
        ('word2sec', word2sec, 2),
        ('sec2tokenMap', sec2tokenMap, 4),
        ('sec2token', sec2token, 2),
    ])

    index = {
        'version': VERSION,
        'counts': {'words': n_words, 'sections': n_secs},
        'words': words_arr,
        'punct': punct_arr,
        'tokenizer': {'blendChars': blend_chars, 'extraChars': extra_chars},
        'bin': bin_desc,
    }

    js = _dump(index)
    if save:
        u.write_file(_static_path(b, Resource.INDEX_JSON), js)
        u.write_file_b(_static_path(b, Resource.INDEX_BIN), blob)
    return js, blob


## sections

def _ordered_sections(b: BaseBuilder) -> list[Section]:
    secs = []
    seen = set()

    def walk(sid):
        sec = b.sectionMap.get(sid)
        if not sec or sid in seen:
            return
        seen.add(sid)
        secs.append(sec)
        for sub in sec.subSids or []:
            walk(sub)

    walk('/')
    for sid in sorted(b.sectionMap):
        walk(sid)
    return secs


def _section_text(sec: Section) -> str:
    parts = []

    def walk(el):
        if el.text:
            parts.append(el.text)
            return
        if el.children:
            for c in el.children:
                walk(c)
            parts.append('\n')

    for node in sec.nodes or []:
        if isinstance(node, MarkdownNode):
            walk(node.el)

    return ' '.join(parts).strip()


## tokenizer

def _tokenize(text: str, word_pat: str, blend_pat: str):
    items = []
    matches = set()
    pos = 0

    for m in re.finditer(word_pat, text):
        if m.start() > pos:
            items.append(('g', text[pos:m.start()]))

        w = m.group(0)
        parts = []
        for k, seg in enumerate(re.split(blend_pat, w)):
            if not seg:
                continue
            if k % 2 == 0:
                items.append(('w', seg))
                parts.append(seg)
            else:
                items.append(('g', seg))

        matches.update(parts)
        if len(parts) > 1:
            matches.add(w)
        pos = m.end()

    if pos < len(text):
        items.append(('g', text[pos:]))

    return items, matches


## binary

def _pack_regions(regions: list) -> tuple[bytes, dict]:
    blob = bytearray()
    desc = {}
    for name, seq, size in regions:
        blob += b'\x00' * ((-len(blob)) % size)
        desc[name] = {'off': len(blob), 'count': len(seq), 'type': f'u{size * 8}'}
        blob += _pack(seq, size)
    return bytes(blob), desc


def _pack(seq, size: int) -> bytearray:
    s = struct.Struct('<' + ('I' if size == 4 else 'H'))
    buf = bytearray(len(seq) * size)
    for i, n in enumerate(seq):
        s.pack_into(buf, i * size, n)
    return buf


## io

def _static_path(b: BaseBuilder, fname: str) -> str:
    return os.path.join(b.options.outputDir, b.options.staticDir, fname)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
