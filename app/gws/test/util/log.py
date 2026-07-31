"""Log buffer."""

_buf = []


def write(s):
    _buf.append(s)


def reset():
    _buf.clear()


def get():
    r = list(_buf)
    _buf.clear()
    return r
