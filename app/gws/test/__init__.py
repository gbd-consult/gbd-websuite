"""GWS Test Framework

To run tests in GWS, you need to start a docker-compose environment with the GWS container and some auxiliary containers, and run the test script in the GWS container.

All test functionality is invoked via ``make.sh test`` commands.

There are two ways to run the tests:

- automatically, with ``make.sh test go``. This configures the docker-compose environment for testing, runs tests, and shuts down.
- manually, where you first start the environment with ``make.sh test start`` and then invoke tests with ``make.sh test run``.


In the latter case, it's convenient to start the compose environment in the foreground and invoke ``test run`` in another shell.

When testing manually, you can also select specific tests with the ``--only`` option and provide additional pytest options. See ``make.sh test -h`` for details.


Configuration
-------------

On your machine, you need a dedicated directory where the tester stores its stuff ("base_dir").

The configuration is in "test.ini" in the application root directory. If you need custom options (e.g., a custom base directory path), create a secondary "ini" file with your overrides and pass it as ``make.sh test --ini myconfig.ini``.

Writing tests
-------------

All test files must end with ``_test.py``. All test functions must start with ``test_``. It is recommended to always import the test utilities library, which provides some useful shortcuts and mocks. Here is an example of a test file::

    '''Testing the foo package.'''

    import gws
    import gws.lib.foo as foo
    import gws.test.util as u

    def test_one():
        assert foo.bar == 1

    def test_two():
        with u.raises(ValueError):
            foo.blah()

Coverage reports
----------------

Pass the ``--coverage`` option to ``test run`` to create a coverage report. It will be created in ``base_dir/coverage``.
With ``test go``, the coverage report is created automatically.

"""
