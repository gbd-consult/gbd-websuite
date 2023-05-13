let $ = sel => document.querySelector(sel);
let $$ = sel => document.querySelectorAll(sel);

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
            let a = document.createElement('a')
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
    $('body').classList.remove('withSearch');
}

function searchExec(val) {
    if (val.trim().length === 0) {
        $('body').classList.remove('withSearch');
        return;
    }

    $('body').classList.add('withSearch');

    let secs = searchFindSections(val);
    let html = '';
    if (secs) {
        val = encodeURIComponent(val);
        html = '<ul>' + secs.map(sec => `<li><a href="${sec.u}?search=${val}">${sec.h}</a></li>`).join('') + '</ul>';
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
    syncNavigation();
    window.addEventListener('popstate', syncNavigation);
    searchInit();
    addRefMarks();
}

window.addEventListener('load', main);





