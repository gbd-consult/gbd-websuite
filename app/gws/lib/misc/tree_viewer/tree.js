let OBJ_LIST = {}
let OBJ_INDEX = {}

window.addEventListener('load', () => {
    init().then(() => null)
})

async function init() {
    $('#sidebar-body').textContent = 'loading...'

    let s = $('#OBJ_LIST')
    OBJ_LIST = JSON.parse(s.textContent)

    await sleep(100)
    createIndex()
    setupEventListeners()

    await sleep(100)
    drawObjectList(OBJ_LIST)

    await sleep(100)
    navigateFromHash()
}

//

function createIndex() {
    for (let obj of OBJ_LIST) {
        OBJ_INDEX[obj.$] = getStrings(obj).join(' ').toLowerCase()
    }
}

function setupEventListeners() {
    $('#sidebar-search-box input').addEventListener('input', e => onSidebarSearchInput(e))
    $('#sidebar-search-box button').addEventListener('click', e => onSidebarSearchClear(e))

    window.addEventListener('hashchange', onHashChange)
}

function onHashChange() {
    navigateFromHash()
}

let searchTimer = null

function onSidebarSearchInput(e) {
    clearTimeout(searchTimer)
    searchTimer = setTimeout(runSearch, 500)
}

function onSidebarSearchClear(e) {
    clearTimeout(searchTimer)
    $('#sidebar-search-box input').value = ''
    drawObjectList(OBJ_LIST)
}

//

function runSearch() {
    let words = $('#sidebar-search-box input').value.toLowerCase().match(/\S+/g)

    if (!words) {
        return
    }

    let a = [], b = []

    for (let obj of OBJ_LIST) {
        let ref = obj.$.toLowerCase()
        let idx = OBJ_INDEX[obj.$]

        if (words.some(w => ref.includes(w))) {
            a.push(obj)
            continue
        }
        if (words.some(w => idx.includes(w))) {
            b.push(obj)
        }
    }

    drawObjectList(a.concat(b))
}

function drawObjectList(ls) {
    let html = []

    for (let obj of ls) {
        let ref = obj.$
        let s = ref.replace(/(uid=)([\w.]+)/, '$1<b>$2</b>')
        html.push(`<a href="#${ref}">${s}</a>`)
    }

    $('#sidebar-body').innerHTML = html.join('')

}

function navigateFromHash() {
    let obj = getObject(location.hash.slice(1))
    if (obj) {
        navigateToObject(obj)
    } else {
        $('#main').innerHTML = ''
    }
}

function navigateToObject(obj) {
    JsonViewerExt.create(obj, $('#main'), {navBar: true, depth: 3, arrayHints: true})
}

function getObject(ref) {
    for (let obj of OBJ_LIST) {
        if (obj.$ === ref) {
            return obj
        }
    }
}

function getStrings(obj) {
    let res = []

    JSON.stringify(obj, (key, value) => {
        if (typeof value === 'string' || typeof value === 'number') {
            res.push(value)
        }
        return value
    })

    return res
}

class JsonViewerExt extends JsonViewer {
    formatValue(val) {
        if (typeof val === 'string' && val.startsWith('$.')) {
            let ref = val.split(' @')[0]
            val = val.replace(/(uid=)([\w.]+)/, '$1<b>$2</b>')
            return [true, `<a href="#${ref}">${val}</a>`]
        }
        return super.formatValue(val)
    }
}

function $(sel, parEl) {
    return (parEl || document).querySelector(sel)
}

function $add(tag, parEl) {
    let el = document.createElement(tag)
    if (!parEl) {
        parEl = document.body
    }
    parEl.appendChild(el)
    return el
}

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}
