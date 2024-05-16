let $ = sel => document.querySelector(sel);
let $$ = sel => document.querySelectorAll(sel);
let $new = tag => document.createElement(tag);

function makeNavigation() {
    let toc = GLOBAL_TOC;

    for (let node of Object.values(toc)) {
        node.open = false;
        node.active = false;
    }

    let u = location.pathname + location.hash;
    u = u.split('?')[0];

    for (let [sid, node] of Object.entries(toc)) {
        if (node.u === u) {
            setActiveNavNode(toc, sid);
            break;
        }
    }

    toc['/'].open = true;


    let li = makeNavNode(toc, '/')
    let div = $('#sidebar-toc');

    while (div.firstChild) {
        div.removeChild(div.firstChild)
    }

    div.appendChild(li.lastChild);
}

let toggleOpen = e => e.target.parentNode.parentNode.classList.toggle('open');

function makeNavNode(toc, sid) {
    let node = toc[sid];

    let li = $new('li');
    li.dataset['sid'] = sid;

    let span = $new('span');
    li.appendChild(span);

    let button = $new('button');
    span.appendChild(button);

    let sub = node.s.map(subSid => makeNavNode(toc, subSid))

    if (node.active) {
        li.classList.add('active');
    }
    if (node.open) {
        li.classList.add('open');
    }
    if (sub.length > 0) {
        li.classList.add('branch');
        button.addEventListener('click', toggleOpen);
    }

    let a = $new('a');
    a.textContent = node.h;
    a.href = node.u;
    span.appendChild(a);

    if (sub.length > 0) {
        let ul = $new('ul');
        for (let s of sub) {
            ul.appendChild(s);
        }
        li.appendChild(ul);
    }

    return li;
}

function setActiveNavNode(toc, sid) {
    let node = toc[sid];
    let parent =  toc[node.p];

    $('#nav-arrow-prev').href = $('#nav-arrow-up').href = $('#nav-arrow-next').href = '#';
    $('#nav-arrow-prev').className = $('#nav-arrow-up').className = $('#nav-arrow-next').className = 'disabled';

    if (!parent) {
        return;
    }

    $('#nav-arrow-up').href = parent.u;
    $('#nav-arrow-up').className = '';

    let i = parent.s.indexOf(sid);

    let prev = toc[parent.s[i - 1]];
    if (prev) {
        $('#nav-arrow-prev').href = prev.u;
        $('#nav-arrow-prev').className = '';
    }

    let next = toc[parent.s[i + 1]];
    if (next) {
        $('#nav-arrow-next').href = next.u;
        $('#nav-arrow-next').className = '';
    }

    node.active = true;
    while (node) {
        node.open = true;
        node = toc[node.p];
    }
}

function syncNavigation() {
    let curr = null;

    $$('#sidebar-toc a').forEach(a => {
        if (a.href === location.href) {
            curr = a
        }
    })

    $$('#sidebar-toc *').forEach(li =>
        li.classList.remove('on')
    )

    if (curr) {

        while (curr.id !== 'sidebar-toc') {
            curr.classList.add('on')
            curr = curr.parentNode
        }
    } else {
        $('#sidebar-toc ul').classList.add('on')
    }
}

function addRefMarks() {
    for (let h of '123456') {
        $$('h' + h).forEach(el => {
            let a = $new('a')
            a.className = 'header-link'
            a.href = el.getAttribute('data-url')
            a.innerHTML = '&para;'
            el.appendChild(a)
        })
    }
}

//

let searchTimer = 0;

function searchInit() {
    $('#sidebar-search input').addEventListener('input', searchRun);
    $('#sidebar-search button').addEventListener('click', searchReset);

    let params = new URLSearchParams('?' + document.location.href.split('?')[1]);
    let val = params && params.get('search');
    if (val) {
        $('#sidebar-search input').value = val;
        searchExec(val);
    }
}

function searchRun(evt) {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => searchExec(evt.target.value), 500);
}

function searchReset() {
    clearTimeout(searchTimer);
    $('#sidebar-search input').value = '';
    $('body').classList.remove('with-search-found', 'with-search-not-found');
    $('#sidebar-search-results').innerHTML = '';
}

function searchExec(val) {
    if (val.trim().length === 0) {
        searchReset();
        return;
    }

    let secs = searchFindSections(val);
    let html = '';

    if (secs) {
        $('body').classList.add('with-search-found');
        $('body').classList.remove('with-search-not-found');
        val = encodeURIComponent(val);
        html = '<ul>' + secs.map(sec => `<li><a href="${sec.u}?search=${val}">${sec.h}</a></li>`).join('') + '</ul>';
    } else {
        $('body').classList.remove('with-search-found');
        $('body').classList.add('with-search-not-found');
        html = '';
    }

    $('#sidebar-search-results').innerHTML = html;
}

const SEARCH_MAX_RESULTS = 50;

function searchFindSections(val) {
    if (!SEARCH_INDEX)
        return;

    let words = val.toLowerCase().match(/[a-zA-ZÄÖÜßäöü_-]+/g);
    if (!words)
        return;

    SEARCH_INDEX._words = SEARCH_INDEX._words || SEARCH_INDEX.words.split('.');
    let indexes = words
        .map(w => SEARCH_INDEX._words.indexOf(w))
        .filter(n => n > 0)
        .map(n => '.' + n.toString(36) + '.');
    if (indexes.length === 0)
        return;

    let exactPhrase = indexes.join('');
    let secs;

    secs = SEARCH_INDEX.sections.filter(sec => sec.w.includes(exactPhrase));
    if (secs.length > 0) {
        return secs.slice(0, SEARCH_MAX_RESULTS);
    }

    secs = SEARCH_INDEX.sections.filter(sec => indexes.every(ix => sec.w.includes(ix)));
    if (secs.length > 0) {
        return secs.slice(0, SEARCH_MAX_RESULTS);
    }
}


//

function main() {
    $('#sidebar-toggle').addEventListener('click', () => {
        document.body.classList.toggle('mobile-sidebar')
    });
    makeNavigation();
    window.addEventListener('popstate', () => {
        makeNavigation();
        document.body.classList.remove('mobile-sidebar')
    });
    searchInit();
    addRefMarks();
}

window.addEventListener('load', main);
