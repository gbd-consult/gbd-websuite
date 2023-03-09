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

function main() {
    syncNavigation()
    window.addEventListener('popstate', syncNavigation)

    for (let h of '123456') {
        document.querySelectorAll('h' + h).forEach(el => {
            let a = document.createElement('a')
            a.className = 'header-link'
            a.href = el.getAttribute('data-url')
            a.innerHTML = '&para;'
            el.appendChild(a)
        })
    }
}

window.addEventListener('load', main);





