Testing
=======

GWS tests are located in ``gws/_test``. Tests are organized in suites, one suite for each server extension or a module. Besides, there is the ``unit`` suite which contains unit tests. Each suite runs in its own docker container.

The structure of a suite is as follows:

- ``data/`` - contains the configuration and resource files for the docker machine
- ``data/config.cx`` - configuration file
- ``init.py`` - (optional) runs in the container before starting the gws server
- ``test*.py`` - one or more test files starting with ``test`` that contain the actual tests

The tests are orchestrated by the command script ``_test/cmd.py`` and a small server that starts and stops docker container on demand. ``cmd.py`` requires some configuration, which is located in ``_test/cmd.ini``, additionally, you can provide a ``--config`` param to the runner if you need to overwrite values from ``cmd.ini``.

The first step is to start the server ::

        cmd.py server --config my.config.ini

Leave the server running on the background and (in a new console) run the selected suite ::

        cmd.py run --config my.config.ini --suite my.suite.name

Run ``cmd.py`` without params to see all the options.

Test files
----------

Our tests run on ``pytest``. Everything that starts with ``test_`` is being run and ``assert`` is used for test assertions. We provide a few helpers in ``util.py`` which you can include in your tests to simplify repetitive tasks.

Fixtures
--------

``make_features`` in ``util.py`` can be used to generate point or square features on a rectangular grid and save them in a postgres table or a geojson file. Additionally, ``common/const.py`` contains a list of reference locations to visually assess positioning and reprojection on a map. Example of use ::

    u.make_features(
        'postgres:my_table',          # create a postgres table my_table
        geom_type='square',           # use square geometries
        prop_schema={                 # generate properties according to this schema
            'property1': 'int',
            'property2': 'text',
        },
        crs='EPSG:3857',              # geometries are in this crs
        rows=10,                      # create 10 rows
        cols=5,                       # create 5 columns
        xy=cc.POINTS.ny,              # use the 'ny' reference location as a starting point
        gap=100,                      # keep 100m between features
    )

Fixture generation code should be placed in ``init.py`` in the respective suite.
