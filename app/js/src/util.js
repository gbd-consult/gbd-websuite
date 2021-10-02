// GWS utilities
// Version __VERSION__

function _gwsPostRequest(cmd, params, onSuccess, onFailure) {
    let data = {
        params: params || {}
    };

    let xhr = new XMLHttpRequest()
    xhr.open('POST', '/_/' + cmd, true)
    xhr.withCredentials = true;
    xhr.setRequestHeader('Content-type', 'application/json');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            let res = xhr.responseText || '';
            try {
                res = JSON.parse(xhr.responseText)
            } catch(exc) {
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

function gwsLogin(onSuccess, onFailure) {
    let err = document.getElementById('gwsError');

    if (err) {
        err.style.display = 'none';
    }

    let params = {
        username: document.getElementById('gwsUsername').value,
        password: document.getElementById('gwsPassword').value
    };

    onSuccess = onSuccess || function () {
        window.location.reload()
    };

    onFailure = onFailure || function () {
        if (err) {
            err.style.display = 'block';
        }
    };

    _gwsPostRequest('authLogin', params, onSuccess, onFailure);
}

function gwsLogout(onSuccess, onFailure) {
    onSuccess = onSuccess || function () {
        window.location.reload()
    };

    onFailure = onFailure || function () {
    };

    _gwsPostRequest('authLogout', {}, onSuccess, onFailure);
}
