"""Tests for the cli module."""

from unittest import mock
import pytest

import gws
import gws.lib.cli as cli
import gws.test.util as u


def test_cprint(capsys):
    cli.SCRIPT_NAME = ''

    # Test with color
    with mock.patch('sys.stdout.isatty', return_value=True):
        cli.cprint('red', 'test message')
        captured = capsys.readouterr()
        assert '\x1b[31mtest message\x1b[0m' in captured.out

        # Test without color (non-tty)
    with mock.patch('sys.stdout.isatty', return_value=False):
        cli.cprint('red', 'test message')
        captured = capsys.readouterr()
        assert 'test message' in captured.out
        assert '\x1b[31m' not in captured.out

        # Test with script name
    cli.SCRIPT_NAME = 'testscript'
    cli.cprint('blue', 'test message')
    captured = capsys.readouterr()
    assert '[testscript] test message' in captured.out
    cli.SCRIPT_NAME = ''  # Reset for other tests


def test_error_warning_info(capsys):
    cli.error('error message')
    captured = capsys.readouterr()
    assert 'error message' in captured.out

    cli.warning('warning message')
    captured = capsys.readouterr()
    assert 'warning message' in captured.out

    cli.info('info message')
    captured = capsys.readouterr()
    assert 'info message' in captured.out


def test_fatal(capsys):
    with u.raises(SystemExit) as excinfo:
        cli.fatal('fatal message')
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert 'fatal message' in captured.out


def test_run(capsys):
    # Test successful command
    with mock.patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        cli.run('echo test')
        mock_run.assert_called_once()
        captured = capsys.readouterr()
        assert '> echo test' in captured.out

        # Test command as list
    with mock.patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        cli.run(['echo', 'test'])
        mock_run.assert_called_once()
        captured = capsys.readouterr()
        assert '> echo test' in captured.out

        # Test failing command
    with mock.patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 1
        with pytest.raises(SystemExit):
            cli.run('false')
        captured = capsys.readouterr()
        assert 'COMMAND FAILED' in captured.out


def test_exec():
    # Test successful command
    result = cli.exec('echo "hello world"')
    assert result == "hello world"

    # Test failing command
    with mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = Exception("Command failed")
        result = cli.exec('invalid command')
        assert 'FAILED' in result


def test_find_dirs(tmp_path):
    # Create test directory structure
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir2").mkdir()
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / "file1").write_text("test")

    # Test finding directories
    dirs = list(cli.find_dirs(tmp_path))
    assert len(dirs) == 2
    assert str(tmp_path / "dir1") in dirs
    assert str(tmp_path / "dir2") in dirs
    assert str(tmp_path / ".hidden_dir") not in dirs

    # Test with non-existent directory
    dirs = list(cli.find_dirs(tmp_path / "nonexistent"))
    assert len(dirs) == 0


def test_find_files(tmp_path):
    # Create test directory structure
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file1.txt").write_text("test")
    (tmp_path / "dir1" / "file2.py").write_text("test")
    (tmp_path / "dir2").mkdir()
    (tmp_path / "dir2" / "file3.txt").write_text("test")
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / ".hidden_dir" / "file4.txt").write_text("test")
    (tmp_path / "file5.txt").write_text("test")

    # Test finding all files
    files = list(cli.find_files(tmp_path))
    assert len(files) == 4
    assert str(tmp_path / "dir1" / "file1.txt") in files
    assert str(tmp_path / "dir1" / "file2.py") in files
    assert str(tmp_path / "dir2" / "file3.txt") in files
    assert str(tmp_path / "file5.txt") in files

    # Test with pattern
    files = list(cli.find_files(tmp_path, pattern=r'\.txt$'))
    assert len(files) == 3
    assert str(tmp_path / "dir1" / "file1.txt") in files
    assert str(tmp_path / "dir2" / "file3.txt") in files
    assert str(tmp_path / "file5.txt") in files
    assert str(tmp_path / "dir1" / "file2.py") not in files

    # Test without recursion
    files = list(cli.find_files(tmp_path, deep=False))
    assert len(files) == 1
    assert str(tmp_path / "file5.txt") in files

    # Test with non-existent directory
    files = list(cli.find_files(tmp_path / "nonexistent"))
    assert len(files) == 0


def test_read_write_file(tmp_path):
    test_file = tmp_path / "test.txt"

    # Test writing to file
    cli.write_file(str(test_file), "test content")
    assert test_file.read_text() == "test content"

    # Test reading from file
    content = cli.read_file(str(test_file))
    assert content == "test content"


def test_parse_args():
    # Test basic arguments
    args = cli.parse_args(["script.py", "arg1", "arg2"])
    assert args[0] == "script.py"
    assert args[1] == "arg1"
    assert args[2] == "arg2"

    # Test options
    args = cli.parse_args(["script.py", "-o", "value", "--long", "longval"])
    assert args["o"] == "value"
    assert args["long"] == "longval"

    # Test flags
    args = cli.parse_args(["script.py", "-f", "--flag"])
    assert args["f"] is True
    assert args["flag"] is True

    # Test rest arguments
    args = cli.parse_args(["script.py", "-", "rest1", "rest2"])
    assert args["_rest"] == ["rest1", "rest2"]


def test_main():
    # Test help flag
    with mock.patch('sys.argv', ['script.py', '-h']), \
            mock.patch('sys.exit') as mock_exit, \
            mock.patch('builtins.print') as mock_print:
        cli.main("test", lambda x: 0, "Usage: test")
        mock_print.assert_called()
        mock_exit.assert_called_with(0)

        # Test normal execution
    with mock.patch('sys.argv', ['script.py', 'arg']), \
            mock.patch('sys.exit') as mock_exit:
        cli.main("test", lambda x: 42, "Usage: test")
        mock_exit.assert_called_with(42)

        # Test exception handling

    def failing_func(args):
        raise Exception("Test exception")

    with mock.patch('sys.argv', ['script.py', 'arg']), \
            mock.patch('sys.exit') as mock_exit, \
            mock.patch('gws.lib.cli.error') as mock_error:
        cli.main("test", failing_func, "Usage: test")
        mock_error.assert_called()


# def test_text_table():
#     # Test with list of dicts
#     data = [
#         {"name": "Alice", "age": 30},
#         {"name": "Bob", "age": 25}
#     ]
#
#     # With header
#     table = cli.text_table(data, header=["name", "age"])
#     assert "name | age" in table
#     assert "Alice | 30" in table
#
#     # With auto header
#     table = cli.text_table(data, header="auto")
#     assert "name | age" in table
#     assert "Alice | 30" in table
#
#     # Without header
#     table = cli.text_table(data, header=None)
#     assert "name | age" not in table
#     assert "Alice | 30" in table
#
#     # Test with list of lists
#     data = [
#         ["Alice", 30],
#         ["Bob", 25]
#     ]
#
#     table = cli.text_table(data, header=["name", "age"])
#     assert "name | age" in table
#     assert "Alice | 30" in table
#
#     # Test with empty data
#     assert cli.text_table([]) == ""


def test_progress_indicator(capsys):
    # Test initialization and basic logging
    with mock.patch('time.time', side_effect=[0,
                                              1.5]):  # Mock time to get consistent results
        with cli.ProgressIndicator("Test", total=100) as progress:
            captured = capsys.readouterr()
            assert "START" in captured.out
            assert "100" in captured.out

            # Test update
            progress.update(10)
            captured = capsys.readouterr()
            assert captured.out.strip() == "[test] Test: 10%"

            # Test update reaching threshold
            progress.update(40)
            captured = capsys.readouterr()
            assert "50%" in captured.out

            # Test end message
        captured = capsys.readouterr()
        assert "END" in captured.out

        # Test without total
    with cli.ProgressIndicator("NoTotal") as progress:
        captured = capsys.readouterr()
        assert "START" in captured.out

        # Updates should do nothing
        progress.update(50)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""
