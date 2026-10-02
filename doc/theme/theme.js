'use strict';

// The blue theme. All DOM, events and rendering live here; navigation data and search come
// from the Dog client core (window.Dog, see _dog.js).

const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);
const $new = (tag, props=null) => {
    const el = document.createElement(tag);
    if (props) {
        Object.assign(el, props);
    }
    return el;
};

const STATIC = document.documentElement.dataset.static || '';
const SEARCH_DEBOUNCE = 150;
const SEARCH_LIMIT = 50;
const NAV_SCROLL_KEY = 'dog.navScroll';
const SEARCH_TEXT_KEY = 'dog.searchText';
const SIDEBAR_KEY = 'dog.sidebar';
const MAX_DEPTH = 999;

let searchSeq = 0;

async function main() {
    initSidebar();

    const dog = await Dog.load(STATIC);

    buildNav(dog);
    initSearch(dog);
    addRefMarks();
    prepareConfigRef();
    revealNav();

    window.addEventListener('popstate', () => buildNav(dog));
    window.addEventListener('pagehide', saveNavScroll);

    if (location.search.includes('dev=1')) {
        document.body.classList.add('with_dev_mode');
    }
}

function initSidebar() {
    $('#sidebar_toggle').addEventListener('click', () => {
        const open = document.body.classList.toggle('sidebar_open');
        if (matchMedia('(min-width: 768px)').matches) {
            sessionStorage.setItem(SIDEBAR_KEY, open ? 'open' : 'closed');
        }
    });
}

function revealNav() {
    const content = $('#sidebar_content');
    const saved = sessionStorage.getItem(NAV_SCROLL_KEY);
    if (content && saved !== null) {
        content.scrollTop = +saved;
    }
    $('#sidebar_toc').classList.add('ready');
}

function saveNavScroll() {
    const content = $('#sidebar_content');
    if (content) {
        sessionStorage.setItem(NAV_SCROLL_KEY, content.scrollTop);
    }
}

// ---------------------------------------------------------------- navigation

function buildNav(dog) {
    const active = dog.toc.forUrl(location.pathname + location.hash);

    let shown = active;
    let depth = MAX_DEPTH;
    for (const sec of dog.toc.breadcrumbs(active)) {
        depth = Math.min(depth, sec.tocDepth ?? MAX_DEPTH);
        shown = sec;
        if (depth < 2) {
            break;
        }
        depth -= 1;
    }

    const open = new Set();
    for (let sec = shown; sec; sec = sec.parent) {
        open.add(sec);
    }

    // the sidebar shows the root's children, not the root itself
    const rootNode = makeNavNode(dog.toc.root, shown, open, MAX_DEPTH);
    $('#sidebar_toc').replaceChildren(rootNode.lastChild || rootNode);

    setArrows(dog, active);
}

function makeNavNode(sec, active, open, depth) {
    const li = $new('li');
    li.dataset.sid = sec.sid;

    const span = $new('span');
    const button = $new('button');
    span.append(button);

    const a = $new('a', { textContent: sec.title, href: sec.url });
    span.append(a);
    li.append(span);

    if (sec === active) {
        li.classList.add('active');
    }
    if (open.has(sec)) {
        li.classList.add('open');
    }

    depth = Math.min(depth, sec.tocDepth ?? MAX_DEPTH);
    if (sec.children.length && depth > 1) {
        li.classList.add('branch');
        button.addEventListener('click', toggleOpen);
        const ul = $new('ul');
        for (const child of sec.children) {
            ul.append(makeNavNode(child, active, open, depth - 1));
        }
        li.append(ul);
    }

    return li;
}

const toggleOpen = evt => evt.currentTarget.closest('li').classList.toggle('open');

function setArrows(dog, active) {
    const targets = {
        prev: active ? dog.toc.prev(active) : null,
        next: active ? dog.toc.next(active) : null,
    };
    const arrows = {
        prev: $('#nav_arrow_prev'),
        next: $('#nav_arrow_next'),
    };

    for (const [dir, el] of Object.entries(arrows)) {
        const target = targets[dir];
        el.href = target ? target.url : '#';
        el.querySelector('.nav_title').textContent = target ? target.title : '';
        el.classList.toggle('disabled', !target);
    }
}

// ---------------------------------------------------------------- search

function initSearch(dog) {
    const input = $('#search input');
    const clear = $('#search button');
    const results = $('#search_results');
    let timer;

    const showClear = () => clear.classList.toggle('visible', input.value !== '');
    const persist = () => localStorage.setItem(SEARCH_TEXT_KEY, input.value);

    input.value = localStorage.getItem(SEARCH_TEXT_KEY) || '';
    showClear();

    input.addEventListener('input', () => {
        clearTimeout(timer);
        showClear();
        persist();
        if (input.value.trim()) {
            openSearch();     // drop the popup down as soon as the search starts
            timer = setTimeout(() => runSearch(dog, input.value, results), SEARCH_DEBOUNCE);
        } else {
            closeSearch(results);
        }
    });

    input.addEventListener('focus', () => {
        requestAnimationFrame(() => input.select());
        if (input.value.trim()) {
            openSearch();
            runSearch(dog, input.value, results);
        }
    });

    clear.addEventListener('click', () => {
        input.value = '';
        showClear();
        persist();
        closeSearch(results);
        input.focus();
    });

    document.addEventListener('keydown', evt => {
        if (evt.key === 'Escape') {
            input.value = '';
            showClear();
            persist();
            closeSearch(results);
        }
    });

    document.addEventListener('click', evt => {
        if (!evt.target.closest('#search')) {
            closeSearch(results);
        }
    });
}

function openSearch() {
    document.body.classList.add('searching');
}

function closeSearch(results) {
    document.body.classList.remove('searching');
    results.replaceChildren();
}

async function runSearch(dog, text, results) {
    const seq = ++searchSeq;
    const found = await dog.query(text.trim(), { limit: SEARCH_LIMIT, snippetWidth: 1000 });
    if (seq === searchSeq) {
        renderResults(dog, results, found);
    }
}

function renderResults(dog, el, { results, totalCount }) {
    el.replaceChildren();

    if (!results.length) {
        el.append($new('div', {
            className: 'search_result_empty',
            textContent: 'keine Ergebnisse gefunden',
        }));
        return;
    }

    el.append($new('div', {
        className: 'search_result_count',
        textContent: totalCount >= SEARCH_LIMIT
            ? `${SEARCH_LIMIT}+ Ergebnisse`
            : `${totalCount} ${totalCount === 1 ? 'Ergebnis' : 'Ergebnisse'}`,
    }));

    const ul = $new('ul');
    for (const hit of results) {
        ul.append(makeResult(dog, hit));
    }
    el.append(ul);
}

function makeResult(dog, hit) {
    const li = $new('li');

    const a = $new('a', { className: 'search_result', href: hit.section.url });

    const crumbs = dog.toc.breadcrumbs(hit.section);   // drop the first (home) crumb
    a.append($new('div', {
        className: 'search_result_title',
        textContent: crumbs.slice(1).map(sec => sec.title).join(' › ') || crumbs.at(-1).title,
    }));

    const snippet = $new('div', { className: 'search_result_snippet' });
    fillSnippet(snippet, hit.snippet);
    a.append(snippet);

    li.append(a);
    return li;
}

function fillSnippet(el, snippet) {
    if (!snippet.length) {
        return;
    }
    if (!snippet[0].atBegin) {
        el.append('… ');
    }
    for (const frag of snippet) {
        if (frag.marker) {
            const [start, len] = frag.marker;
            if (start) {
                el.append(frag.text.slice(0, start));
            }
            el.append($new('mark', { textContent: frag.text.slice(start, start + len) }));
            const rest = frag.text.slice(start + len);
            if (rest) {
                el.append(rest);
            }
        } else {
            el.append(frag.text);   // text node — auto-escaped
        }
    }
    if (!snippet.at(-1).atEnd) {
        el.append(' …');
    }
}

// ---------------------------------------------------------------- content

function addRefMarks() {
    for (const el of $$('main h1, main h2, main h3, main h4, main h5, main h6')) {
        const url = el.dataset.url;
        if (!url) {
            continue;
        }
        el.append($new('a', { className: 'header_link', href: url, textContent: '¶' }));
    }
}

function prepareConfigRef() {
    for (const marker of $$('main .configref_category_marker')) {
        const cls = marker.classList[1];
        let h = marker.closest('p') || marker;
        while (h && h.tagName !== 'H2') {
            h = h.previousElementSibling;
        }
        if (!h) {
            continue;
        }
        h.classList.add(cls);
    }
}


main();
