Using the command line
======================

Once the server is started, you can use the ``gws`` command line tool (CLI) to control it. ``gws`` runs inside the container, so it's invoked with ``docker exec`` ::


    docker exec -it my-gws-container gws -h

It's recommended to create an alias, so that you can invoke the CLI just as as ``gws ...`` ::

    alias gws='docker exec -it my-gws-container gws'


The general syntax for CLI commands is ::


    gws command subcommand option option...


for example ::


    gws cache seed --layers my-layer --level 3


See :doc:`cliref` for the list of all available commands and subcommands.
