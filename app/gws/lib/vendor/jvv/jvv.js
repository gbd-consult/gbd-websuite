class JsonViewer {

    static create(data, containerEl, options) {
        return new this(data, containerEl, options)
    }

    DEFAULT_OPTIONS = {
        navBar: false,
        arrayHints: false,
        depth: 2,
        arrayLimit: 20_000,
        depthLimit: 1000,
        searchDelay: 500,
        searchLimit: 50,
        searchContextLimit: 100,
    }

    constructor(data, containerEl, options) {
        this.options = Object.assign({}, this.DEFAULT_OPTIONS, options ?? {})

        this.data = data

        this.containerEl = containerEl
        this.mainEl = null
        this.contentEl = null

        this.tickCount = 0
        this.tickUpdateStep = 5000

        this.searchInputEl = null
        this.searchResultsEl = null
        this.searchRe = null
        this.searchTimer = null
        this.searchResults = null
        this.searchResultIndex = null
        this.searchInputs = []

        this.nowait(this.renderAll(this.options.depth))
    }


    //

    onObjectClick(e) {
        if (this.isLocked()) {
            return
        }

        let rowEl = e.currentTarget
        let depth = e.altKey ? this.options.depthLimit : 1
        this.nowait(this.toggleObject(rowEl, depth))
    }

    onExpandAll() {
        if (this.isLocked()) {
            return
        }
        this.nowait(this.renderContent(this.options.depthLimit))
    }

    onCollapseAll() {
        if (this.isLocked()) {
            return
        }
        this.renderContent(1).then(() => null)
    }

    onSelectAll() {
        if (this.isLocked()) {
            return
        }
        this.selectAll()
    }

    onSearchInput(e) {
        clearTimeout(this.searchTimer)
        this.searchTimer = setTimeout(() => this.searchStart(), this.options.searchDelay)
    }

    onSearchReset(e) {
        clearTimeout(this.searchTimer)
        this.removeClass(this.mainEl, 'jvv-has-search-results')
        this.clear(this.searchResultsEl)
        this.searchInputEl.value = ''
    }

    onSearchResultClick(e, rowEl) {
        if (this.isLocked()) {
            return
        }
        this.nowait(this.searchGotoResult(rowEl))
    }

    //

    async searchStart() {
        this.removeClass(this.mainEl, 'jvv-has-search-results')
        this.clear(this.searchResultsEl)

        this.lock()

        let text = this.searchInputEl.value || ''
        text = text.trim()

        if (text.length === 0 || text === ':') {
            this.unlock()
            return
        }

        let keySearch = false

        if (text[0] === ':') {
            keySearch = true
            text = text.slice(1).trim()
        }

        let search = {
            re: new RegExp(this.escapeRe(text), 'i'),
            results: [],
            keySearch: keySearch,
            hasMore: false,
        }

        await this.searchCollect([], this.data, search)
        this.unlock()

        this.searchRenderResults(search)
    }

    async searchCollect(keys, val, search) {
        if (this.searchInputs.length > 0) {
            return
        }

        if (search.results.length >= this.options.searchLimit) {
            search.hasMore = true
            return
        }

        await this.addTick()

        let t = this.getType(val)

        switch (t) {
            case this.T.bigint:
            case this.T.number:
            case this.T.string:
                let res = this.searchCheckValue(keys, val, search)
                if (res) {
                    search.results.push(res)
                }
                return
            case this.T.array:
            case this.T.object:
                let [isArray, iter, len] = this.getObjectProps(val)
                for (let [k, v] of iter) {
                    await this.searchCollect(keys.concat([k]), v, search)
                }
        }
    }

    searchCheckValue(keys, val, search) {
        let hk = this.keysToPath(keys)
        let hv = String(val)

        if (search.keySearch) {
            hk = hk.replace(search.re, '\x01$&\x02')
            if (!hk.includes('\x01')) {
                return
            }
            hv = this.shorten(hv, this.options.searchContextLimit, false)
        } else {
            hv = hv.replace(search.re, '\x01$&\x02')
            if (!hv.includes('\x01')) {
                return
            }
            hv = hv.replace(/(.+?)\x01(.+?)\x02(.+)/, (_, m1, m2, m3) => {
                m1 = this.shorten(m1, this.options.searchContextLimit, true)
                m3 = this.shorten(m3, this.options.searchContextLimit, false)
                return m1 + '\x01' + m2 + '\x02' + m3
            })
        }

        return {
            htmlKeys: this.escapeHTMLWithMarks(hk),
            htmlVal: this.escapeHTMLWithMarks(hv),
            keys: keys,
        }


    }

    searchRenderResults(search) {
        this.removeClass(this.mainEl, 'jvv-has-search-results')
        this.clear(this.searchResultsEl)

        if (search.results.length === 0) {
            return
        }

        this.addClass(this.mainEl, 'jvv-has-search-results')

        for (let res of search.results) {
            let rowEl = this.add(this.searchResultsEl, this.span('jvv-search-result'))
            rowEl.dataset.keys = JSON.stringify(res.keys)
            this.add(rowEl, this.htmlSpan('jvv-key', res.htmlKeys + ': '))
            this.add(rowEl, this.htmlSpan('jvv-value', res.htmlVal))
            rowEl.addEventListener('click', e => this.onSearchResultClick(e, rowEl))
        }

        let total = String(search.results.length)
        if (search.hasMore) {
            total += '+'
        }
        this.add(this.searchResultsEl, this.span('jvv-search-total', total))
    }



    async searchGotoResult(rowEl) {

        await this.renderContent(1)

        let bodyEl = this.objectBodyElement(this.contentEl.firstChild)
        if (!bodyEl) {
            return
        }

        let keys = JSON.parse(rowEl.dataset.keys)
        rowEl = await this.expandChain(bodyEl, keys)
        if (rowEl) {
            rowEl.scrollIntoView({ behavior: 'auto', block: 'center', inline: 'nearest' })
            this.addClass(rowEl, 'jvv-search-highlight')
            setTimeout(() => this.removeClass(rowEl, 'jvv-search-highlight'), 1000)
        }
    }

    //

    async expandChain(bodyEl, foundKeys) {
        for (let rowEl of bodyEl.querySelectorAll('.jvv-row')) {
            if (!rowEl.dataset || !rowEl.dataset.keys) {
                continue
            }

            let keys = JSON.parse(rowEl.dataset.keys)
            let c = this.compareArrays(foundKeys, keys)

            if (c === 2) {
                return rowEl
            }
            if (c === 1 && rowEl.classList.contains('jvv-object')) {
                await this.renderObjectBody(rowEl, 1)
                return await this.expandChain(this.objectBodyElement(rowEl), foundKeys)
            }
        }
    }

    async toggleObject(rowEl, depth) {
        if (rowEl.classList.contains('jvv-expanded')) {
            await this.unrenderObjectBody(rowEl)
            this.removeClass(rowEl, 'jvv-expanded')
            return
        }

        this.lock()
        await this.renderObjectBody(rowEl, depth)
        this.unlock()
    }

    selectAll() {
        let range = document.createRange()
        range.selectNodeContents(this.contentEl)
        let sel = window.getSelection()
        sel.removeAllRanges()
        sel.addRange(range)
    }

    //

    isLocked() {
        return this.mainEl && this.mainEl.classList.contains('jvv-locked')
    }

    lock() {
        this.addClass(this.mainEl, 'jvv-locked')
        this.tickCount = 0
    }

    unlock() {
        this.removeClass(this.mainEl, 'jvv-locked')
    }

    //

    async renderAll(depth) {
        this.clear(this.containerEl)
        this.mainEl = this.add(this.containerEl, this.div('jvv'))
        if (this.options.navBar) {
            await this.renderNav()
        }
        this.contentEl = this.add(this.mainEl, this.div('jvv-content'))
        await this.renderContent(depth)
    }

    async renderContent(depth) {
        this.clear(this.contentEl)
        this.lock()
        await this.renderValue(this.contentEl, [], this.data, true, depth)
        this.unlock()
    }

    renderNav() {
        let nav = this.add(this.mainEl, this.div('jvv-nav'))

        let navRow = this.add(nav, this.div('jvv-nav-row'))

        this.renderNavButton(navRow, 'Expand All', 'jvv-expand-all-button', e => this.onExpandAll(e))
        this.renderNavButton(navRow, 'Collapse All', 'jvv-collapse-all-button', e => this.onCollapseAll(e))
        this.renderNavButton(navRow, 'Select All', 'jvv-select-all-button', e => this.onSelectAll())

        let searchBox = this.add(navRow, this.span('jvv-search-box'))

        this.searchInputEl = this.add(searchBox, this.elem('input', 'jvv-search-input'))
        this.searchInputEl.addEventListener('input', e => this.onSearchInput(e))
        this.searchInputEl.value = ''

        this.renderNavButton(searchBox, 'Reset', 'jvv-search-reset-button', e => this.onSearchReset(e))


        this.searchResultsEl = this.add(nav, this.elem('div', 'jvv-search-results'))
    }

    renderNavButton(parEl, title, cls, handler) {
        let b = this.add(parEl, this.span('jvv-nav-button ' + cls))
        b.title = title
        b.addEventListener('click', handler)
    }

    async renderValue(parEl, keys, val, isLast, depth) {
        await this.addTick()

        let t = this.getType(val)

        switch (t) {
            case this.T.boolean:
            case this.T.number:
            case this.T.bigint:
            case this.T.string:
                return this.renderPrimitive(parEl, keys, val, isLast)
            case this.T.emptyarray:
            case this.T.emptyobject:
            case this.T.function:
            case this.T.null:
            case this.T.symbol:
            case this.T.undefined:
                return this.renderSpecial(parEl, keys, this.valueT[t], isLast)
            case this.T.array:
            case this.T.object:
                return this.renderObject(parEl, keys, val, isLast, depth)
        }
    }

    renderSpecial(parEl, keys, val, isLast) {
        let cls = 'jvv-value jvv-type-special'
        let valueEl = this.span(cls, val)
        this.drawSimpleRow(parEl, keys, valueEl, isLast)
    }

    renderPrimitive(parEl, keys, val, isLast) {
        let t = this.getType(val)
        let cls = 'jvv-value  jvv-type-' + this.nameT[t]
        let [isHTML, f] = this.formatValue(val)
        let valueEl = isHTML ? this.htmlSpan(cls, f) : this.span(cls, f)
        console.log({ val, f, valueEl })
        this.drawSimpleRow(parEl, keys, valueEl, isLast)
    }

    formatValue(val) {
        return [false, JSON.stringify(val)]
    }

    async renderObject(parEl, keys, val, isLast, depth) {
        let rowEl = this.drawRow(parEl, keys)
        this.add(parEl, this.span('jvv-object-body'))

        let [isArray, iter, len] = this.getObjectProps(val)

        this.addClass(rowEl, 'jvv-object')

        if (isArray) {
            this.addClass(rowEl, 'jvv-array')
            if (this.options.arrayHints) {
                rowEl.firstChild.dataset.after = '(' + len + ') '
            }
        }

        let ob = (isArray ? '[' : '{')
        this.add(rowEl, this.drawPunct(ob))

        rowEl.addEventListener('click', e => this.onObjectClick(e, rowEl))

        if (depth > 0) {
            await this.renderObjectBody(rowEl, depth)
        }

        let cb = (isArray ? ']' : '}') + (isLast ? '' : ',')
        this.add(parEl, this.drawPunctRow(cb))
    }

    async renderObjectBody(rowEl, depth) {
        this.addClass(rowEl, 'jvv-expanded')

        let bodyEl = this.objectBodyElement(rowEl)
        if (!bodyEl) {
            return
        }
        this.clear(bodyEl)

        let keys = JSON.parse(rowEl.dataset.keys)
        let val = this.getObject(keys)

        let [isArray, iter, len] = this.getObjectProps(val)
        let innerEl = this.add(bodyEl, this.span('jvv-object-body-inner'))

        let n = 0
        for (let [k, v] of iter) {
            n += 1
            if (n > this.options.arrayLimit) {
                this.add(innerEl, this.drawMore(len - n + 1))
                break
            }
            await this.renderValue(innerEl, keys.concat([k]), v, n === len, depth - 1)
        }
    }

    async unrenderObjectBody(rowEl) {
        let bodyEl = this.objectBodyElement(rowEl)
        this.clear(bodyEl)
    }

    objectBodyElement(rowEl) {
        if (!rowEl) {
            return
        }
        let bodyEl = rowEl.nextSibling
        if (bodyEl && bodyEl.classList.contains('jvv-object-body')) {
            return bodyEl
        }
    }

    //

    drawRow(parEl, keys) {
        let rowEl = this.add(parEl, this.span('jvv-row'))
        rowEl.dataset.keys = JSON.stringify(keys)

        let key = keys.length > 0 ? keys[keys.length - 1] : null;
        this.add(rowEl, this.drawKey(key))

        return rowEl
    }

    drawSimpleRow(parEl, keys, valueEl, isLast) {
        let rowEl = this.drawRow(parEl, keys)

        this.add(rowEl, valueEl)

        if (!isLast) {
            this.add(rowEl, this.drawPunct(','))
        }
    }

    drawKey(key) {
        let keyEl = this.span('jvv-key')
        let t = typeof key

        if (t === 'number' && this.options.arrayHints) {
            keyEl.dataset.before = '[' + key + '] '
        } else if (t === 'string') {
            keyEl.textContent = JSON.stringify(key) + ': '
        }
        return keyEl
    }

    drawPunct(text, title) {
        return this.span('jvv-punct', text)
    }

    drawPunctRow(text, title) {
        return this.span('jvv-punct-row', text)
    }

    drawMore(count) {
        return this.span('jvv-more', null, '…(' + count + ')')
    }

    //

    shorten(s, limit, fromEnd) {
        if (s.length <= limit) {
            return s
        }
        if (fromEnd) {
            return '…' + s.slice(-limit).trimLeft()
        } else {
            return s.slice(0, limit).trimRight() + '…'
        }
    }

    span(cls, text) {
        return this.elem('span', cls, text)
    }

    htmlSpan(cls, html) {
        return this.elem('span', cls, null, html)
    }

    div(cls) {
        return this.elem('div', cls)
    }

    elem(tag, cls, text, html) {
        let el = document.createElement(tag)

        if (cls) {
            el.className = cls
        }

        if (text) {
            el.textContent = text
        }

        if (html) {
            el.innerHTML = html
        }

        return el
    }

    add(parEl, el) {
        parEl.appendChild(el)
        return el
    }

    addClass(el, cls) {
        el.classList.add(cls)
        return el
    }

    removeClass(el, cls) {
        el.classList.remove(cls)
        return el
    }

    clear(el) {
        if (el) {
            el.innerHTML = ''
        }
        return el
    }

    //

    T = {
        array: 1,
        bigint: 2,
        boolean: 3,
        emptyarray: 4,
        emptyobject: 5,
        function: 6,
        null: 7,
        number: 8,
        object: 9,
        string: 10,
        symbol: 11,
        undefined: 12,
    }

    typeofT = {
        'bigint': this.T.bigint,
        'boolean': this.T.boolean,
        'function': this.T.function,
        'number': this.T.number,
        'string': this.T.string,
        'symbol': this.T.symbol,
        'undefined': this.T.undefined,
    }

    nameT = {
        [this.T.array]: 'array',
        [this.T.bigint]: 'bigint',
        [this.T.boolean]: 'boolean',
        [this.T.emptyarray]: 'emptyarray',
        [this.T.emptyobject]: 'emptyobject',
        [this.T.function]: 'function',
        [this.T.null]: 'null',
        [this.T.number]: 'number',
        [this.T.object]: 'object',
        [this.T.string]: 'string',
        [this.T.symbol]: 'symbol',
        [this.T.undefined]: 'undefined',
    }

    valueT = {
        [this.T.emptyarray]: '[]',
        [this.T.emptyobject]: '{}',
        [this.T.function]: '"<function>"',
        [this.T.null]: 'null',
        [this.T.symbol]: '"<symbol>"',
        [this.T.undefined]: '"<undefined>"',
    }

    getType(val) {
        let t = typeof val

        if (this.typeofT[t]) {
            return this.typeofT[t]
        }

        if (!val) {
            return this.T.null
        }

        if (Array.isArray(val)) {
            return (val.length === 0) ? this.T.emptyarray : this.T.array
        }

        for (let _ in val) {
            return this.T.object
        }

        return this.T.emptyobject
    }

    getObject(keys) {
        let val = this.data
        for (let k of keys) {
            val = val[k]
        }
        return val
    }

    getObjectProps(val) {
        let isArray = Array.isArray(val)
        let iter = isArray ? val.entries() : Object.entries(val)
        let len = isArray ? val.length : iter.length
        return [isArray, iter, len]
    }

    keysToPath(keys) {
        let a = []
        for (let k of keys) {
            if (typeof k === 'number') {
                a.push('[' + k + ']')
            } else {
                if (a.length > 0) {
                    a.push('.')
                }
                a.push(k)
            }
        }
        return a.join('')
    }

    compareArrays(arr, sub) {
        if (sub.length > arr.length) {
            return -1
        }
        for (let i = 0; i < sub.length; i++) {
            if (arr[i] !== sub[i]) {
                return -1
            }
        }
        return sub.length === arr.length ? 2 : 1
    }

    escapeRe(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    }

    escapeHTML(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    }

    escapeHTMLWithMarks(str) {
        str = this.escapeHTML(str)
        str = str.replace(/\x01/g, '<mark>')
        str = str.replace(/\x02/g, '</mark>')
        return str
    }

    async addTick() {
        this.tickCount = ((this.tickCount || 0) + 1) % (this.tickUpdateStep)
        if (this.tickCount === 0) {
            await this.sleep(1)
        }
    }


    sleep(n) {
        return new Promise(res => setTimeout(res, n))
    }

    nowait(p) {
        p.then(() => null)
    }

}
