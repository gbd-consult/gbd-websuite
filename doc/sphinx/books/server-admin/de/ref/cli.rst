Referenz der Kommandozeilen-Tools
=================================

Sobald der GBD WebSuite Server gestartet ist, steht Ihnen das Kommandozeilen-Tool (CLI) ``gws`` zur Verfügung. ``gws`` läuft innerhalb des GWS Containers, also wird es mit ``docker exec`` aufgerufen::

    docker exec -it my-gws-container gws -h

wobei ``my-gws-container`` ein von Ihnen vergebener Container-Name ist.

Sie können für diesen Befehl einen Alias oder Skript erstellen, z.B. ::

    alias gws='docker exec -it my-gws-container gws'

Die allgemeine Syntax für CLI-Befehle lautet ::

    gws command subcommand option option...

zum Beispiel ::

    gws cache seed --layers my-layer --level 3

Nachfolgend finden Sie eine Liste aller Befehle und deren Optionen.

.. include:: ../../../../ref/de.cliref.txt
