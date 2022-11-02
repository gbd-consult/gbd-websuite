let $ = sel => document.querySelector(sel);
let $$ = sel => document.querySelectorAll(sel);

function syncNavigation() {
    let curr = null;

    $$('#sidebar-toc a').forEach(a => {
        if (a.href === location.href) {
            curr = a
        }
    })

    if (!curr)
        return;

    $$('#sidebar-toc *').forEach(li =>
        li.classList.remove('on')
    )

    while (curr.id !== 'sidebar-toc') {
        curr.classList.add('on')
        curr = curr.parentNode;
    }


    console.log(location)


}

function main() {

    syncNavigation()
    window.addEventListener('popstate', syncNavigation)
}

window.addEventListener('load', main);





