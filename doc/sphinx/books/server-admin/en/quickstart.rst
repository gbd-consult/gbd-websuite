Quick start
===========

This section shows how to run GBD WebSuite server for the first time and set up a first project.

**Step 1**. Make sure you have `docker <https://www.docker.com>`_ installed and working.

**Step 2**. Download and run the GBD WebSuite server image ::

    docker run -it -p 3333:80 --name my-gws-container gbdconsult/gws-server:latest

This will run GBD WebSuite server on port ``3333`` under the name ``my-gws-container`` (feel free to use another name and/or port). On some setups, you might need root rights (``sudo``) to be able to run this command.


If everything is right, you should see the server log on your terminal. Point your browser to `<http://localhost:3333>`_. There will be the server start page with our sample project.

Now, stop the server with Control-C and remove the container::

    docker rm my-gws-container

**Step 3**. Create a directory ``hello`` somewhere on your hard drive (e.g. ``/var/work/hello``).
Create a file named ``config.json`` in that directory, with the following content ::


    {
        "title": "Hello",
        "map": {
            "view": {
                "extent": [554000, 6461000, 954000, 6861000],
                "scales": [1e3, 5e3, 1e4]
            },
            "layers": [
                {
                    "title": "OpenStreetMap",
                    "type": "client",
                    "kind": "osm"
                }
            ]
        }
    }


**Step 4**. Run the container again, this time giving it a path to your newly created configuration ::

    docker run -it -p 3333:80 --name my-gws-container --mount type=bind,src=/var/work/hello,dst=/data/projects/hello gbdconsult/gws-server:latest

Navigate to `<http://localhost:3333/hello>`_. You should see the Open Street Map of Düsseldorf, the birth place of GWS.
