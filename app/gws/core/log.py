"""Logging facility"""

import codecs
import logging
import logging.config
import sys

from . import error as err
from . import util

_gws_logger_name = 'gws'


class Level:
    CRITICAL = 50
    ERROR = 40
    WARN = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0
    ALL = 0


def _caller_info(skip_frames):
    try:
        frame = getattr(sys, '_getframe')()
    except:
        frame = sys.exc_info()[2].tb_frame

    if not frame:
        return ['?', 0, '?']

    while 1:
        co = getattr(frame, 'f_code', None)
        filename = co.co_filename if co else ''
        if filename != __file__ and 'logging' not in filename:
            break
        back = getattr(frame, 'f_back', None)
        if not back or back == frame:
            break
        frame = back

    while skip_frames > 0:
        back = getattr(frame, 'f_back', None)
        if not back or back == frame:
            break
        frame = back
        skip_frames -= 1

    co = getattr(frame, 'f_code', None)
    return [
        co.co_filename if co else '',
        getattr(frame, 'f_lineno', 0),
        co.co_name if co else ''
    ]


# borrowed from logging.py
# added support for skip_frames, and avoid the costly caller lookup unless debugging

class _Logger(logging.Logger):
    def exception(self, msg='', *args, **kwargs):
        _, exc, _ = sys.exc_info()
        pfx = 'EXCEPTION:: '

        self.fatal(pfx + (msg or repr(exc)))

        if self.isEnabledFor(Level.DEBUG):
            for k in err.string().split('\n'):
                self.debug(pfx + k)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if _logger.disabled:
            return

        if level == logging.DEBUG and self.name != _gws_logger_name:
            # disable 3rd party debug logging - there's really too much
            return

        filename, lineno, func = '', 0, ''

        if self.isEnabledFor(Level.DEBUG):
            skip_frames = 0

            try:
                skip_frames = extra.pop('skip_frames')
            except (AttributeError, KeyError):
                pass

            filename, lineno, func = _caller_info(skip_frames)

        if exc_info and not isinstance(exc_info, tuple):
            exc_info = sys.exc_info()

        record = self.makeRecord(self.name, level, filename, lineno, msg, args, exc_info, func, extra)
        self.handle(record)


class _Formatter(logging.Formatter):
    def format(self, r):
        try:
            msg = r.msg % r.args
        except TypeError:
            msg = r.msg
            if not isinstance(msg, str):
                msg = repr(msg)
            if r.args:
                msg = msg + ': ' + ' '.join(repr(a) for a in r.args)

        # ts = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(r.created))
        tn = str(r.threadName or '').replace('uWSGIWorker', '').replace('Core', '/').replace('MainThread', '0')

        return (
                'GWS['
                + str(r.process)
                + '/'
                + (tn or '')
                + (' %s:%d' % (r.pathname, r.lineno) if r.pathname else '')
                + '] ' + r.levelname + ':: '
                + msg
        )


def _config(name):
    return {
        'version': 1,
        'formatters': {
            name: {'()': _Formatter}
        },

        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'stream': codecs.getwriter('ascii')(sys.stdout.buffer, 'backslashreplace'),
                'formatter': name,
                'level': 0
            }
        },

        'root': {
            'level': 0,
            'handlers': [name],
        },

        'loggers': {
            name: {
                'level': 0,
                'handlers': [name],
                'propagate': False
            }
        }
    }


def _init():
    logging.setLoggerClass(_Logger)
    logging.config.dictConfig(_config(_gws_logger_name))
    return logging.getLogger(_gws_logger_name)


_logger = util.get_app_global('gws.logger', _init)

fatal = _logger.fatal
warn = _logger.warn
info = _logger.info
error = _logger.error
exception = _logger.exception
debug = _logger.debug

_levels = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
    'ALL': logging.NOTSET,
}


def set_level(level):
    level = _levels.get(level, level or logging.WARNING)
    logging.disable(max(0, level - 1))


def enable():
    _logger.disabled = False


def disable():
    _logger.disabled = True


def is_debug():
    return _logger.isEnabledFor(Level.DEBUG)