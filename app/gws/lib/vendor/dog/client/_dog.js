// Dog client core — the browser counterpart of dog/indexer.py.
//
// Owns the on-wire format (the _toc.json / _index.json / _index.bin artifacts, the CSR
// binary layout, the tokenizer and the Index algorithm) and exposes it as data. It touches
// no DOM; a theme consumes this and does all rendering, events and popups.
//
//   const dog = await Dog.load(_baseUrl);   // fetches _toc.json; index is deferred
//   dog.toc                                 // navigation, ready immediately
//   await dog.query('postgres', { limit, snippetWidth });
//
// Dog.create(tocJson, indexJson, arrayBuffer) builds an instance from already-fetched data
// (no fetch) — used by tests.

(function (window) {
    "use strict";

    const PUNCT_MAX = 100; // token ids 1..PUNCT_MAX are punctuation; above are words
    const SNIPPET_WIDTH = 200; // default snippet size, in characters
    const MAX_RESULTS = 50;
    const MATCH_MODE = "all";

    const Resource = {
        TOC_JSON: "_toc.json",
        INDEX_JSON: "_index.json",
        INDEX_BIN: "_index.bin",
    };

    // ================================================================ helpers

    // Escape characters for literal use inside a regex character class.
    const _escapeClass = (chars) => chars.replace(/[-\\\]^]/g, "\\$&");

    // Strip a URL of its query string / hash for exact and file-level lookups.
    const _normUrl = (u) => u.split("?")[0];
    const _baseUrl = (u) => _normUrl(u).split("#")[0];

    // Range of integers [lo, hi).
    const _range = (lo, hi) => {
        let a = [];
        for (let i = lo; i < hi; i++) {
            a.push(i);
        }
        return a;
    };

    // Return a unique array of values, preserving order.
    const _uniq = (a) => Array.from(new Set(a));

    // Toc
    // The section tree from _toc.json: lookups, breadcrumbs and reading-order navigation.

    class Toc {
        _byNid = {}; // nid -> section
        _bySid = {}; // sid -> section
        _byUrl = {}; // exact url (query stripped) -> section
        _byBase = {}; // file path (url without hash) -> the section that owns the file
        _pages = []; // file-owning sections in document (reading) order

        constructor(json) {
            this.sections = json.sections;
            this.root = null;

            for (const sec of this.sections) {
                this._byNid[sec.nid] = sec;
                this._bySid[sec.sid] = sec;
                this._byUrl[_normUrl(sec.url)] = sec;

                const base = _baseUrl(sec.url);
                if (!(base in this._byBase)) {
                    this._byBase[base] = sec;
                }
            }

            for (const sec of this.sections) {
                sec.parent = sec.parent != null ? this._byNid[sec.parent] : null;
                sec.children = sec.children.map((nid) => this._byNid[nid]);
                if (!sec.parent) {
                    this.root = sec;
                }
            }

            this._pages = this.sections.filter((sec) => sec === this._byBase[_baseUrl(sec.url)]);
        }

        // Section by numeric id, or null.
        byNid(nid) {
            return this._byNid[nid] || null;
        }

        // Section by sid, or null.
        bySid(sid) {
            return this._bySid[sid] || null;
        }

        // Section whose url matches exactly (query stripped), or null.
        byUrl(u) {
            return this._byUrl[_normUrl(u)] || null;
        }

        // Active section for a page url: exact match, else the section owning the file, else null.
        forUrl(u) {
            u = _normUrl(u);
            return this._byUrl[u] || this._byBase[_baseUrl(u)] || null;
        }

        // Ancestor chain, root -> sec.
        breadcrumbs(sec) {
            const out = [];
            while (sec) {
                out.unshift(sec);
                sec = sec.parent;
            }
            return out;
        }

        // The page (file-owning section) that sec is rendered on.
        _pageOf(sec) {
            return this._byBase[_baseUrl(sec.url)] || null;
        }

        // Previous page in reading order (the file before sec's), or null.
        prev(sec) {
            const i = this._pages.indexOf(this._pageOf(sec));
            return i > 0 ? this._pages[i - 1] : null;
        }

        // Next page in reading order (the file after sec's), or null.
        next(sec) {
            const i = this._pages.indexOf(this._pageOf(sec));
            return i >= 0 && i < this._pages.length - 1 ? this._pages[i + 1] : null;
        }
    }

    // Index
    // The inverted index and token streams from _index.json + _index.bin.

    class Index {
        _lowerWords; // words lowercased, for the prefix bsearch
        _dog; // backref to the Dog instance, for toc lookups

        constructor(dog, json, buffer) {
            this._dog = dog;

            this.words = json.words; // 0-based surfaces, sorted by lowercased form
            this.punct = json.punct; // 0-based punctuation strings

            const extra = json.tokenizer.extraChars ? _escapeClass(json.tokenizer.extraChars) : "";
            const wc = "[\\p{L}\\p{N}" + extra + "]+";
            if (json.tokenizer.blendChars) {
                const bc = "[" + _escapeClass(json.tokenizer.blendChars) + "]+";
                this.wordRe = new RegExp(wc + "(?:" + bc + wc + ")*", "gu");
                this.blendRe = new RegExp(bc, "gu");
            } else {
                this.wordRe = new RegExp(wc, "gu");
                this.blendRe = null;
            }

            const view = (d) =>
                d.type === "u32" ? new Uint32Array(buffer, d.off, d.count) : new Uint16Array(buffer, d.off, d.count);

            this.word2secMap = view(json.bin.word2secMap);
            this.word2sec = view(json.bin.word2sec);
            this.sec2tokenMap = view(json.bin.sec2tokenMap);
            this.sec2token = view(json.bin.sec2token);

            this._lowerWords = this.words.map((w) => w.toLowerCase());
        }

        query(text, limit, width) {
            const terms = this._parse(text);
            if (!terms.length) {
                return { results: [], totalCount: 0 };
            }

            const mwi = [];
            const hitsForNid = new Map();
            for (const term of terms) {
                const nidsForTerm = [];
                for (const wid of this._wordIdsForTerm(term)) {
                    mwi.push(wid);
                    nidsForTerm.push(...this._secNidsForWordId(wid));
                }
                for (const nid of _uniq(nidsForTerm)) {
                    hitsForNid.set(nid, (hitsForNid.get(nid) || 0) + 1);
                }
            }
            const matchedWordIds = _uniq(mwi);

            const minHits = MATCH_MODE === "all" ? terms.length : 1;
            const resultNids = Array.from(hitsForNid.keys())
                .filter((nid) => hitsForNid.get(nid) >= minHits)
                .sort((a, b) => hitsForNid.get(b) - hitsForNid.get(a) || a - b);

            const results = [];
            for (const nid of resultNids.slice(0, limit)) {
                const section = this._dog.toc.byNid(nid);
                const snippet = this._snippet(section, terms, matchedWordIds, width);
                results.push({ section, snippet });
            }

            return { results, totalCount: resultNids.length };
        }

        // Tokenize query text into lowercased prefix terms: each word's fine-grained parts plus,
        // for a blend run, the joined form. Mirrors the index-time tokenizer so query and index agree.
        _parse(text) {
            const terms = [];

            for (const m of text.matchAll(this.wordRe)) {
                const t = m[0].toLowerCase();
                terms.push(t);

                if (this.blendRe) {
                    const parts = t.split(this.blendRe).filter(Boolean);
                    terms.push(...parts);
                }
            }

            return _uniq(terms);
        }

        // Section nids containing a word id.
        _secNidsForWordId(wid) {
            const a = this.word2secMap[wid];
            const z = this.word2secMap[wid + 1];

            return _uniq(_range(a, z).map((i) => this.word2sec[i]));
        }

        // Token ids for a section nid.
        _tokenIdsForSecNid(nid) {
            const a = this.sec2tokenMap[nid];
            const z = this.sec2tokenMap[nid + 1];

            return _range(a, z).map((k) => this.sec2token[k]);
        }

        // Word ids (1-based indices into `words`) for words that start with term.
        _wordIdsForTerm(term) {
            const lo = this._findWordId(term);
            const hi = this._findWordId(term + "\uFFFF"); // max code unit; appended to bracket the prefix range

            return _range(lo, hi);
        }

        // First word id whose lowercased form is >= term.
        _findWordId(term) {
            let lo = 0;
            let hi = this.words.length;

            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (this._lowerWords[mid] < term) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }

        _snippet(section, terms, matchedWordIds, width) {
            const termsByLen = [...terms].sort((a, b) => b.length - a.length);

            const fragments = [];
            let firstPos = -1;

            const tokenIds = this._tokenIdsForSecNid(section.nid);
            const matchedTokenIds = new Set(matchedWordIds.map((wid) => wid + PUNCT_MAX));

            for (let [pos, tid] of tokenIds.entries()) {
                let marker = null;
                let text = null;
                let hit = false;

                if (tid <= PUNCT_MAX) {
                    text = this.punct[tid];
                } else {
                    text = this.words[tid - PUNCT_MAX];
                    if (matchedTokenIds.has(tid)) {
                        hit = true;
                        marker = this._marker(text, termsByLen);
                    }
                }

                const isWord = tid > PUNCT_MAX;

                if (isWord && fragments.length > 0 && fragments.at(-1).isWord) {
                    fragments.push({
                        text: " ",
                        marker: null,
                        isWord: false,
                        atBegin: false,
                        atEnd: false,
                    });
                }

                if (hit && firstPos < 0) {
                    firstPos = fragments.length;
                }

                fragments.push({
                    text,
                    marker,
                    isWord,
                    atBegin: pos === 0,
                    atEnd: pos === tokenIds.length - 1,
                });
            }

            if (firstPos < 0) {
                return fragments;
            }

            const pre = [];
            const post = [];

            for (let i = firstPos - 1, w = width >> 2; i >= 0; i--) {
                const f = fragments[i];
                pre.unshift(f);
                w -= f.text.length;
                if (f.isWord && w <= 0) {
                    break;
                }
            }
            for (let i = firstPos + 1, w = width >> 2; i < fragments.length; i++) {
                const f = fragments[i];
                post.push(f);
                w -= f.text.length;
                if (f.isWord && w <= 0) {
                    break;
                }
            }

            return [...pre, fragments[firstPos], ...post];
        }

        _marker(text, termsByLen) {
            const lower = text.toLowerCase();
            for (const term of termsByLen) {
                if (lower.startsWith(term)) {
                    return [0, term.length];
                }
            }
        }
    }

    // Main
    // Ties the toc and (lazily loaded) Index index together.

    class Dog {
        constructor(toc, index = null, baseDir = null) {
            this.toc = toc;
            this.index = index;
            this.baseDir = baseDir;
        }

        // Build from already-fetched data, no network — used by tests.
        static create(tocJson, indexJson, buffer) {
            const dog = new Dog(new Toc(tocJson));
            if (indexJson && buffer) {
                dog.index = new Index(dog, indexJson, buffer);
            }
            return dog;
        }

        // Load _toc.json eagerly; the Index index is fetched lazily on the first query.
        static async load(baseDir = "") {
            const tocJson = await fetch(baseDir + "/" + Resource.TOC_JSON).then((r) => r.json());
            return new Dog(new Toc(tocJson), null, baseDir);
        }

        // Search: resolves to { results: [{ section, snippet }] ranked (at most options.limit),
        // totalCount: how many sections matched in all }.
        async query(text, options = {}) {
            if (!this.index) {
                const [indexJson, buffer] = await Promise.all([
                    fetch(this.baseDir + "/" + Resource.INDEX_JSON).then((r) => r.json()),
                    fetch(this.baseDir + "/" + Resource.INDEX_BIN).then((r) => r.arrayBuffer()),
                ]);
                this.index = new Index(this, indexJson, buffer);
            }

            return this.index.query(text, options.limit || MAX_RESULTS, options.snippetWidth || SNIPPET_WIDTH);
        }
    }

    window.Dog = Dog;
})(window);
