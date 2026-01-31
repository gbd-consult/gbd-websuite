let $ = (sel, parent) => (parent || document).querySelector(sel);
let $$ = (sel, parent) => [...(parent || document).querySelectorAll(sel)];

window.addEventListener("lxxoad", function () {
    $('#button-login-submit').addEventListener('click', evt => {
        gwsLogin()
        evt.preventDefault();
    });

    $('#button-logout-submit').addEventListener('click', evt => {
        gwsLogout()
        evt.preventDefault();
    });

  });
