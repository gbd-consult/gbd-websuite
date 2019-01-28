// gws start helper script

function _gwsRequest(cmd, params, success, fail) {
    var data = {
        cmd: cmd,
        params: params || {}
    };

    var xhr = new XMLHttpRequest()
    xhr.open('POST', '/_', true)
    xhr.withCredentials = true;
    xhr.setRequestHeader('Content-type', 'application/json');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                success();
            } else {
                fail();
            }
        }
    };

    xhr.send(JSON.stringify(data))
}

function gwsLogin(success, fail) {
    var err = document.getElementById('gwsError');

    if (err) {
        err.style.display = 'none';
    }

    var params = {
        username: document.getElementById('gwsUsername').value,
        password: document.getElementById('gwsPassword').value
    };

    success = success || function () {
        window.location.reload()
    };

    fail = fail || function () {
        if (err) {
            err.style.display = 'block';
        }
    };

    _gwsRequest('authLogin', params, success, fail);
}

function gwsLogout(success, fail) {
    success = success || function () {
        window.location.reload()
    };

    fail = fail || function () {
    };

    _gwsRequest('authLogout', {}, success, fail);
}
