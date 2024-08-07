// GWS utilities
// Version __VERSION__

window.addEventListener("load", function (evt) {
    let frmLogin = document.getElementById('gwsLoginForm');
    if (frmLogin) {
        frmLogin.addEventListener('submit', function(evt) {
            gwsLogin();
            evt.preventDefault();
            return false;
        })
    }

    let frmLogout = document.getElementById('gwsLogoutForm');
    if (frmLogout) {
        frmLogout.addEventListener('submit', function(evt) {
            gwsLogout();
            evt.preventDefault();
            return false;
        })
    }
});

function gwsLogin(onSuccess, onFailure) {
    let cls = document.body.classList;

    cls.remove('gwsLoginError');
    cls.add('gwsLoginProgress');

    let params = {
        username: document.getElementById('gwsUsername').value,
        password: document.getElementById('gwsPassword').value
    };

    onSuccess = onSuccess || (() => window.location.reload());
    onFailure = onFailure || (() => 0);

    _gwsPostRequest(
        'authLogin',
        params,
        () => {
            cls.remove('gwsLoginProgress');
            onSuccess()
        },
        () => {
            cls.remove('gwsLoginProgress');
            cls.add('gwsLoginError');
            onFailure()
        },
    );
}

function gwsLogout(onSuccess, onFailure) {
    onSuccess = onSuccess || (() => window.location.reload());
    onFailure = onFailure || (() => 0);

    _gwsPostRequest('authLogout', {}, onSuccess, onFailure);
}

function _gwsPostRequest(cmd, params, onSuccess, onFailure) {
    let data = params || {};

    let xhr = new XMLHttpRequest()
    xhr.open('POST', '/_/' + cmd, true)
    xhr.withCredentials = true;
    xhr.setRequestHeader('Content-type', 'application/json');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            let res = xhr.responseText || '';
            try {
                res = JSON.parse(xhr.responseText)
            } catch (exc) {
                res = {'text': res}
            }
            if (xhr.status === 200) {
                onSuccess(xhr.status, res);
            } else {
                onFailure(xhr.status, res);
            }
        }
    };

    xhr.send(JSON.stringify(data))
}
