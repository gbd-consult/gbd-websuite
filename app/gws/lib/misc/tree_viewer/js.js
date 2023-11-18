let $ = (s, par) => (par || document).querySelector(s);
let $new = s => document.createElement(s);

window.addEventListener('load', init);

function init() {
    let main = $new('div');
    main.id = 'main';
    $('body').appendChild(main);
    setTimeout(init2, 0);
}

function init2() {
    addColumn(''); // object list

    let h = location.hash;
    if (h)
        navigateToObject(h.slice(1), $('#main').lastChild);

    $('body').addEventListener('click', function (evt) {
        // handle <a rel=> navigation
        let h = evt.target.rel || evt.target.parentElement.rel;
        if (h)
            navigateToObject(h.split(' @ ')[0], getColumn(evt.target));
    });

}

function navigateToObject(ref, srcColumn) {
    let main = $('#main');

    if (ref === 'root') {
        location.hash = '';
        while (main.children.length > 1)
            main.removeChild(main.lastChild);
        return;
    }

    location.hash = ref;

    while (main.lastChild && main.lastChild !== srcColumn)
        main.removeChild(main.lastChild);

    if (main.lastChild && main.lastChild.OBJECT_REF === ref)
        main.removeChild(main.lastChild);

    addColumn(ref);
}

function addColumn(ref) {
    let col = $new('div');

    col.className = 'column';
    col.innerHTML = `
        <div class="column-top">
            <input>
            <button ${ref ? '' : 'disabled'}>&times;</button>
        </div>
        <div class="column-content"></div>
        <div class="column-bottom">
            <a rel="${ref || 'root'}">${ref ? ref : PATH}</a>
        </div>
    `;

    $('input', col).addEventListener('input', evt =>
        updateColumnText(getColumn(evt.target))
    );
    $('button', col).addEventListener('click', evt =>
        $("#main").removeChild(getColumn(evt.target))
    );

    $('#main').appendChild(col);
    $('#main').scrollLeft = 9999999;

    col.OBJECT_REF = ref;
    updateColumnText(col);
}

function updateColumnText(col) {
    let ref = col.OBJECT_REF;
    let obj = getObject(ref);
    let flt = ($('input', col).value || '');

    if (flt) {
        if (!ref)
            obj = DATA.filter(x => matches(flt, x)).map(x => x.$)
        else if (Array.isArray(obj))
            obj = obj.filter(x => matches(flt, x))
        else
            obj = Object.fromEntries(
                Object.entries(obj).filter(kv => matches(flt, kv))
            )
    }
    $('.column-content', col).innerHTML = JSON.stringify(formatObject(obj), null, 4);
}

function getColumn(el) {
    while (1) {
        if (!el)
            return addColumn('');
        if (el.parentNode && el.parentNode.id === 'main')
            return el;
        el = el.parentNode;
    }
}

function getObject(ref) {
    if (!ref)
        return DATA.map(x => x.$);
    return DATA.find(x => x.$ === ref) || `object [${ref}] not found`;
}

function formatObject(obj) {
    if (typeof obj === 'string')
        return formatString(obj, false);

    if (!obj || typeof obj !== 'object')
        return obj;

    if (Array.isArray(obj))
        return obj.map(x => formatObject(x));

    let r = {};
    for (let [k, v] of Object.entries(obj))
        r[formatString(k, true)] = formatObject(v);
    return r;
}

function formatString(val, isKey) {
    let rel = (!isKey && val.startsWith('$.')) ? val : null;
    let html = htmlize(val);

    html = html.replace(/(uid=\S+)/, '<u>$1</u>');

    if (rel)
        html = `<a rel='${rel}'>${html}</a>`;
    if (isKey)
        html = `<b>${html}</b>`;

    return html;
}

function matches(search, obj) {
    if (search[0] === '*')
        return JSON.stringify(obj).includes(search.slice(1))
    return (obj.$ || '').includes(search)
}

function htmlize(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

