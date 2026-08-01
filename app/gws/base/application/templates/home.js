// GWS home page utilities

window.addEventListener('load', function (evt) {
    let frmLogin = document.getElementById('gws-home-login-form');
    if (frmLogin) {
        frmLogin.addEventListener('submit',  (evt) => {
            gwsLogin(frmLogin.action);
            evt.preventDefault();
            return false;
        });
    }

    let frmLogout = document.getElementById('gws-home-logout-form');
    if (frmLogout) {
        frmLogout.addEventListener('submit',  (evt) => {
            gwsLogout(frmLogout.action);
            evt.preventDefault();
            return false;
        });
    }
});

function gwsLogin(actionUrl, onSuccess, onFailure) {
    let cls = document.body.classList;

    cls.remove('gws-home-login-error');
    cls.add('gws-home-login-progress');

    let params = {
        username: document.getElementById('gws-home-username').value,
        password: document.getElementById('gws-home-password').value,
        to: (new URLSearchParams(window.location.search).get('to') || '').trim(),
    };

    onSuccess =
        onSuccess ||
        ((status, res) => {
            if (res.redirectTo) {
                window.location.href = res.redirectTo;
            } else {
                window.location.reload();
            }
        });
    onFailure = onFailure || (() => 0);

    _gwsPostRequest(
        actionUrl,
        params,
        (status, res) => {
            cls.remove('gws-home-login-progress');
            res = res || {};
            res.redirectTo = _gwsValidRedirectUrl(res.redirectTo);
            onSuccess(status, res);
        },
        (status, res) => {
            cls.remove('gws-home-login-progress');
            cls.add('gws-home-login-error');
            onFailure(status, res);
        },
    );
}

function _gwsValidRedirectUrl(s) {
    if (!s) {
        return '';
    }
    try {
        let u = new URL(s, window.location.origin);
        return u.origin === window.location.origin ? u.pathname + u.search + u.hash : '';
    } catch (exc) {
        return '';
    }
}

function gwsLogout(actionUrl, onSuccess, onFailure) {
    onSuccess = onSuccess || (() => window.location.reload());
    onFailure = onFailure || (() => 0);

    _gwsPostRequest(actionUrl, {}, onSuccess, onFailure);
}

function _gwsPostRequest(actionUrl, params, onSuccess, onFailure) {
    let data = params || {};

    let xhr = new XMLHttpRequest();
    xhr.open('POST', actionUrl, true);
    xhr.withCredentials = true;
    xhr.setRequestHeader('Content-type', 'application/json');

    xhr.onreadystatechange =  () => {
        if (xhr.readyState === 4) {
            let res = xhr.responseText || '';
            try {
                res = JSON.parse(xhr.responseText);
            } catch (exc) {
                res = { text: res };
            }
            if (xhr.status === 200) {
                onSuccess(xhr.status, res);
            } else {
                onFailure(xhr.status, res);
            }
        }
    };

    xhr.send(JSON.stringify(data));
}
