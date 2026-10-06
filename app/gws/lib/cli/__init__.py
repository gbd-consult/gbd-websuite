"""Utilities for command line scripts.

This package is used by the ``gws`` command line commands and by the
development and test scripts. It has no dependencies on the rest of GWS and
can be imported outside of the GWS application.

It provides:

- colored console output: ``cprint``, ``info``, ``warning``, ``error`` and
  ``fatal``, which exits the script. Messages are prefixed with
  ``SCRIPT_NAME`` when set, colors are used only on a terminal.
- shell commands: ``run`` echoes and runs a command and exits on failure,
  ``exec`` runs a command and returns its output.
- simple file utilities: ``find_dirs``, ``find_files``, ``ensure_dir``,
  ``read_file``, ``write_file``.
- a script entry point: ``parse_args`` parses ``-opt value`` and
  ``--opt value`` style arguments, ``main`` runs a main function with them
  and prints the usage on ``-h``.
- ``text_table``, which formats rows as a plain text table.
- ``ProgressIndicator``, a context manager that logs the progress of a
  long-running task in percent steps.

Example::

    import gws.lib.cli as cli

    USAGE = '''
    Usage: myscript.py <dir> [-pattern <regex>]
    '''

    def main(args):
        paths = list(cli.find_files(args[1], args.get('pattern')))
        cli.info(cli.text_table([{'path': p} for p in paths], header='auto'))
        with cli.ProgressIndicator('processing', len(paths)) as pi:
            for p in paths:
                pi.update()
        return 0

    if __name__ == '__main__':
        cli.main('myscript', main, USAGE)
"""

import re
import os
import sys
import subprocess
import time
import math
import traceback

SCRIPT_NAME = ''
"""Name of the running script, used as a prefix for messages."""

_COLOR = {
    'black': '\x1b[30m',
    'red': '\x1b[31m',
    'green': '\x1b[32m',
    'yellow': '\x1b[33m',
    'blue': '\x1b[34m',
    'magenta': '\x1b[35m',
    'cyan': '\x1b[36m',
    'white': '\x1b[37m',
    'reset': '\x1b[0m',
}


def cprint(clr, msg):
    """Print a message to stdout, in color if stdout is a terminal.

    Args:
        clr: Color name, e.g. ``red`` or ``cyan``, or an empty value for no color.
        msg: Message.
    """
    if SCRIPT_NAME:
        msg = '[' + SCRIPT_NAME + '] ' + msg
    if clr and sys.stdout.isatty():
        msg = _COLOR[clr] + msg + _COLOR['reset']
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def error(msg):
    """Print an error message in red.

    Args:
        msg: Message.
    """
    cprint('red', msg)


def fatal(msg):
    """Print an error message in red and exit with code 1.

    Args:
        msg: Message.
    """
    cprint('red', msg)
    sys.exit(1)


def warning(msg):
    """Print a warning message in yellow.

    Args:
        msg: Message.
    """
    cprint('yellow', msg)


def info(msg):
    """Print an info message in cyan.

    Args:
        msg: Message.
    """
    cprint('cyan', msg)


##

def run(cmd):
    """Print and run a shell command, exit the script if it fails.

    The command output is not captured.

    Args:
        cmd: Command as a string or a list of strings, which are joined with spaces.
    """
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    cmd = re.sub(r'\s+', ' ', cmd.strip())
    info(f'> {cmd}')
    res = subprocess.run(cmd, shell=True, capture_output=False)
    if res.returncode:
        fatal(f'COMMAND FAILED, code {res.returncode}')


def exec(cmd):
    """Run a shell command and return its output.

    Args:
        cmd: Command string.

    Returns:
        The stripped stdout of the command, or an error message if the command could not be run.
    """
    try:
        return (
            subprocess
            .run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            .stdout.decode('utf8').strip()
        )
    except Exception as exc:
        return f'> {cmd} FAILED: {exc}'


def find_dirs(dirname):
    """Find the subdirectories of a directory, not recursively.

    Hidden directories are skipped.

    Args:
        dirname: Directory path.

    Yields:
        Subdirectory paths. Nothing if ``dirname`` is not a directory.
    """
    if not os.path.isdir(dirname):
        return

    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue
        if de.is_dir():
            yield de.path


def find_files(dirname, pattern=None, deep=True):
    """Find files in a directory.

    Hidden files and directories are skipped.

    Args:
        dirname: Directory path.
        pattern: Regular expression to search for in file paths.
        deep: Search subdirectories too.

    Yields:
        File paths. Nothing if ``dirname`` is not a directory.
    """
    if not os.path.isdir(dirname):
        return

    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue
        if de.is_dir() and deep:
            yield from find_files(de.path, pattern)
            continue
        if de.is_file() and (pattern is None or re.search(pattern, de.path)):
            yield de.path


def ensure_dir(path):
    """Create a directory, including parent directories.

    Args:
        path: Directory path.

    Returns:
        The path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def read_file(path):
    """Read a text file.

    Args:
        path: File path.

    Returns:
        The file content, stripped.
    """
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read().strip()


def write_file(path, text):
    """Write a text file.

    Args:
        path: File path.
        text: Content.
    """
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def parse_args(argv):
    """Parse command line arguments.

    ``-opt`` and ``--opt`` set the option ``opt`` to ``True``, a following non-option argument
    sets it to that value instead. Other arguments are stored under integer keys, in order.
    A ``-`` argument stores all remaining arguments as a list under ``_rest``.

    Args:
        argv: Arguments, usually ``sys.argv``.

    Returns:
        A dict of options and positional arguments.
    """
    args = {}
    opt = None
    n = 0

    for a in argv:
        if a == '-':
            args['_rest'] = []
        elif '_rest' in args:
            args['_rest'].append(a)
        elif a.startswith('--'):
            opt = a[2:]
            args[opt] = True
        elif a.startswith('-'):
            opt = a[1:]
            args[opt] = True
        elif opt:
            args[opt] = a
            opt = None
        else:
            args[n] = a
            n += 1

    return args


def main(name, main_fn, usage):
    """Run the main function of a script.

    Parses ``sys.argv`` and calls ``main_fn`` with the parsed arguments. With ``-h`` or ``--help``,
    prints the usage text and exits. The return value of ``main_fn`` is used as the exit code.
    Exceptions are printed as internal errors and exit with code 1, keyboard interrupts exit with code 130.

    Args:
        name: Script name, used as a prefix for messages.
        main_fn: Main function, called with the dict from ``parse_args``.
        usage: Usage text.
    """
    global SCRIPT_NAME

    SCRIPT_NAME = name

    args = parse_args(sys.argv)
    if not args or 'h' in args or 'help' in args:
        print('\n' + usage.strip() + '\n')
        sys.exit(0)

    try:
        sys.exit(main_fn(args))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        error('INTERNAL ERROR')
        error(traceback.format_exc())
        sys.exit(1)


def text_table(data, header=None, delim=' | '):
    """Format rows as a plain text table.

    Numbers are right-aligned, other values left-aligned.

    Args:
        data: Rows, either dicts or sequences.
        header: Column keys, or ``auto`` to use the keys (or indexes) of the first row.
            If given, a header line is printed. If ``None``, the columns of the first row are used without a header.
        delim: Column delimiter.

    Returns:
        The table text, or an empty string if there are no rows.
    """

    data = list(data)

    if not data:
        return ''

    is_dict = isinstance(data[0], dict)

    print_header = header is not None
    if header is None or header == 'auto':
        header = data[0].keys() if is_dict else list(range(len(data[0])))

    widths = [len(h) if print_header else 1 for h in header]

    def get(d, h):
        if is_dict:
            return d.get(h, '')
        try:
            return d[h]
        except IndexError:
            return ''

    for d in data:
        widths = [
            max(a, b)
            for a, b in zip(
                widths,
                [len(str(get(d, h))) for h in header]
            )
        ]

    def field(n, v):
        if isinstance(v, (int, float)):
            return str(v).rjust(widths[n])
        return str(v).ljust(widths[n])

    rows = []

    if print_header:
        hdr = delim.join(field(n, h) for n, h in enumerate(header))
        rows.append(hdr)
        rows.append('-' * len(hdr))

    for d in data:
        rows.append(delim.join(field(n, get(d, h)) for n, h in enumerate(header)))

    return '\n'.join(rows)


class ProgressIndicator:
    """Context manager that logs the progress of a task.

    Logs ``START`` on enter, the progress in percent steps on ``update``,
    and ``END`` with the elapsed time on a normal exit.
    """

    def __init__(self, title, total=0, resolution=10):
        """Create a progress indicator.

        Args:
            title: Title, used as a prefix for messages.
            total: Total number of items. If 0, no progress is logged.
            resolution: Step in percent between progress messages.
        """
        self.resolution = resolution
        self.title = title
        self.total = total
        self.progress = 0
        self.lastd = 0
        self.starttime = 0

    def __enter__(self):
        self.log(f'START ({self.total})' if self.total else 'START')
        self.starttime = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            ts = time.time() - self.starttime
            self.log(f'END ({ts:.2f} sec)')

    def update(self, add=1):
        """Add processed items and log the progress if it reached the next step.

        Args:
            add: Number of processed items.
        """
        if not self.total:
            return
        self.progress += add
        p = math.floor(self.progress * 100.0 / self.total)
        if p > 100:
            p = 100
        d = round(p / self.resolution) * self.resolution
        if d > self.lastd:
            self.log(f'{d}%')
        self.lastd = d

    def log(self, s):
        """Log a message with the title.

        Args:
            s: Message.
        """
        info(f'{self.title}: {s}')
