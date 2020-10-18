Benutzung der Befehlsliste
==========================

Sobald der GBD WebSuite Server gestartet ist, können Sie das Kommandozeilen-Tool (CLI) verwenden, um ihn zu steuern. ``gws`` läuft innerhalb des Containers, also wird es mit ``docker exec`` aufgerufen::

    docker exec -it my-gws-container gws -h

Es wird empfohlen, einen Alias zu erstellen, damit Sie die CLI genauso aufrufen können wie ``gws . . . ``` ::

    alias gws='docker exec -it my-gws-container gws'

Die allgemeine Syntax für CLI-Befehle lautet ::

    gws command subcommand option option...

zum Beispiel ::

    gws cache seed --layers my-layer --level 3

Siehe ^cliref für die Liste aller verfügbaren Befehle und Unterbefehle
