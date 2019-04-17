Benutzung der Kommandozeile
===========================

Sobald der GBD WebSuite Server gestartet ist, können Sie zur Steuerung das Kommandozeilen-Tool (CLI) verwenden. Innerhalb des Containers läuft ``gws``, welches mit ``docker exec`` aufgerufen wird::

    docker exec -it my-gws-container gws -h

Es wird empfohlen, einen Alias zu erstellen, damit Sie die CLI genauso aufrufen können wie ``gws . . . ``` ::

    alias gws='docker exec -it my-gws-container gws'


Die allgemeine Syntax für CLI-Befehle lautet ::


    gws command subcommand option option...


zum Beispiel ::


    gws cache seed --layers my-layer --level 3


Siehe :doc:`cliref` für die Liste aller verfügbaren Befehle und Unterbefehle
