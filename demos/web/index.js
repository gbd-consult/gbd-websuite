window.addEventListener('load', onLoad);

let $ = (sel, parent) => (parent || document).querySelector(sel);
let $$ = (sel, parent) => [...(parent || document).querySelectorAll(sel)];

let state = {
    projectUid: '',
    keywords: new Set(),
    searchValue: '',
}

function splitWords(str) {
    return str.toLowerCase().match(/\w+/g) || [];
}

function entryMatches(pe, words) {
    if (words.length === 0)
        return true;
    let text = $('.frame-link', pe).textContent.toLowerCase();
    return words.some(w => text.includes(w));
}

function entryHasKeyword(pe, keywords) {
    if (keywords.size === 0)
        return true;
    let tags = new Set($$('mark', pe).map(m => m.textContent));
    return [...keywords].every(kw => tags.has(kw));
}

function setFrame(url) {
    if (!url) {
        $('#demo-container').innerHTML = '';
        return;
    }

    let frame = $('#demo-container').querySelector('iframe');
    if (frame && frame.src && frame.src.indexOf(url) >= 0) {
        return;
    }

    $('#demo-container').innerHTML = `<iframe src="${url}"></iframe>`;
}

function scrollIntoView(el) {
    let par = el.parentElement;

    if (el.offsetTop + el.offsetHeight > par.scrollTop + par.offsetHeight) {
        par.scrollTop = el.offsetTop - par.offsetHeight + el.offsetHeight + 10;
        return;
    }

    if (el.offsetTop < par.scrollTop) {
        par.scrollTop = el.offsetTop - 10;
        return;
    }
}

function update() {
    $$('.project-list-entry').forEach(pe => pe.classList.remove('selected'));
    $$('.project-list-entry').forEach(pe => pe.classList.add('hidden'));
    $$('.project-details-entry').forEach(pd => pd.classList.add('hidden'));

    let words = splitWords(state.searchValue);

    $$('.project-list-entry').forEach(pe => {
        if (entryMatches(pe, words) && entryHasKeyword(pe, state.keywords))
            pe.classList.remove('hidden');
    });

    $$('mark').forEach(m => {
        m.classList.toggle('selected', state.keywords.has(m.textContent))
    });

    if (state.projectUid) {
        setFrame('/project/' + state.projectUid);

        $$('.project-list-entry').forEach(pe => {
            if (pe.dataset.uid === state.projectUid) {
                pe.classList.add('selected');
                scrollIntoView(pe);
            }
        });

        $$('.project-details-entry').forEach(pd => {
            if (pd.dataset.uid === state.projectUid) {
                pd.classList.remove('hidden');
                document.title = $('h2', pd).textContent;
            }
        });
    } else {
        setFrame('');
    }
}

function toggleKeyword(kw) {
    if (state.keywords.has(kw))
        state.keywords.delete(kw);
    else
        state.keywords.add(kw);
    update();
}

function showProject(uid) {
    let url = '/demo/' + uid;
    history.pushState({}, "", url);
    state.projectUid = uid;
    update();
}

function showPrevNextProject(delta) {
    let visible = $$('.project-list-entry').filter(pe => !pe.classList.contains('hidden'));
    let selected = $$('.project-list-entry').filter(pe => pe.classList.contains('selected'));

    if (visible.length === 0) {
        return
    }

    let pos = 0;

    if (selected.length > 0) {
        pos = visible.indexOf(selected[0]);
        if (pos < 0) {
            pos = 0;
        } else {
            pos += delta;
            if (pos < 0) {
                pos = visible.length - 1;
            }
            if (pos >= visible.length) {
                pos = 0;
            }
        }
    }

    showProject(visible[pos].dataset.uid);
}

function updateFromLocation() {
    let url = window.location.pathname || '';
    let m;

    if ((m = url.match(/demo\/(\w+)/))) {
        state.projectUid = m[1];
    }
    if ((m = url.match(/tag\/(\S+)/))) {
        state.keywords = new Set(m[1].split(','));
    }
    update()
}

function onLoad() {

    updateFromLocation();

    $('#side-search').addEventListener('input', evt => {
        state.searchValue = $('#side-search input').value;
        update();
    });

    $('#project-list').addEventListener('click', evt => {
        let src = evt.target;

        if (src.classList.contains('frame-link')) {
            showProject(src.dataset.uid);
            evt.preventDefault();
            return;
        }

        if (src.tagName.toLowerCase() === 'mark') {
            toggleKeyword(src.textContent);
            evt.preventDefault();
            return
        }
    });

    window.addEventListener('popstate', evt => {
        updateFromLocation();
    })

    $('#button-mobile-view').addEventListener('click', evt => {
        $('body').classList.toggle('with-mobile-view');
        evt.preventDefault();
    });

    $('#button-login').addEventListener('click', evt => {
        $('body').classList.toggle('with-login-box');
        evt.preventDefault();
    });

    $('#button-prev-demo').addEventListener('click', evt => {
        showPrevNextProject(-1);
        evt.preventDefault();
    });

    $('#button-next-demo').addEventListener('click', evt => {
        showPrevNextProject(+1);
        evt.preventDefault();
    });

    $('#button-login-submit').addEventListener('click', evt => {
        gwsLogin()
        evt.preventDefault();
    });

    $('#button-logout-submit').addEventListener('click', evt => {
        gwsLogout()
        evt.preventDefault();
    });

}
