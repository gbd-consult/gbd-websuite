Autorisierung
=============

Die Autorisierung der GBD WebSuite ist rollenbasiert. Das bedeutet, dass den einzelnen Benutzern verschiedene Berechtigungen zugeteilt werden können. Wenn sich der Benutzer anmeldet, werden seine Referenzen nacheinander an alle konfigurierten Provider weitergegeben. Wenn ein Provider die Referenzen akzeptiert, gibt er eine Liste der Rollen für diesen Benutzer zurück.

Wenn der Benutzer ein Objekt anfordert, prüft der Server, welche Berechtigungen dem Benutzer zugeteilt sind. Anhand dieser Berechtigungen werden dem Nutzer dann verschiedene Funktionen wie zum Beispiel das Lesen, Schreiben (z. B. beim Bearbeiten) oder Auszuführen (z. B. eine Serveraktion) zur Verfügung gestellt. Wenn es keine expliziten Berechtigungen auf der gewählten Objektebene gibt, wird das übergeordnete Objekt geprüft dann das nächst übergeordnete Objekt und so weiter. Um diese Prüfungen durchzuführen, liest der Server die ``access``-Datei jedes angeforderten Objekts.

Zugangsregelungen
-----------------


``access`` ist eine Liste von ``AccessRule`` Objekten. Jede ``AccessRule`` enthält

- den ``type`` der Regel - "erlauben" oder "verweigern",
- den ``mode`` - "lesen", "schreiben", "ausführen" oder eine Kombination daraus,
- die Liste der ``role`` Namen, auf die sich die Regel bezieht


Mit Hilfe der ``access``-Regeln kann der Algorithmus zur Prüfung der Berechtigungen, formal wie folgt beschrieben werden::


    ## Der Benutzer U fordert einen Berechtigungsmodus P (z. B. "lesen") für ein Objekt O an:

    let currentObject = O
    let userRoles = "roles" of the user U

    loop

        if currentObject has property "access"

            ## Überprüfen Sie die expliziten Zugriffsregeln:

            for each Rule in currentObject.access
                if (Rule.roles contains any of userRoles) and (Rule.mode contains P)
                    if Rule.type is "allow", return Access Granted
                    if Rule.type is "deny",  return Access Denied
                end if
            end for

        end if

        ## An dieser Stelle hat das aktuelle Objekt entweder keine "Zugriffsregeln",
        ## oder keine dieser Regeln passt zu den Rollen des Benutzers.
        ## Überprüfen Sie das übergeordnete Objekt, wenn es existiert.

        if currentObject has a "parent"
            let currentObject = currentObject.parent
            continue loop
        end if

        ## An diesem Punkt haben wir das Wurzelobjekt erreicht
        ## und haben immer noch keine passende Regel gefunden.
        ## Verwenden Sie die Standardregel "Alle Anfragen ablehnen".

        return Access Denied

    end loop

Rollen
----------

Es gibt einige vordefinierte Rollen, die in der GBD WebSuite eine besondere Bedeutung haben:

TABLE
   *guest* ~ Nicht eingeloggter Benutzer
   *user* ~ Jeder eingeloggter Benutzer
   *everyone* ~ Alle Benutzer, eingeloggt und Gäste
   *admin* ~ Administrator und Benutzer die diese Rolle haben, erhalten automatisch Zugriff auf alle Ressourcen
/TABLE

Andernfalls können Sie beliebige Rollennamen vergeben. Beim Erstellen dieser Rollennamen müssen Sie jedoch auf die Schreibweise achten. Der Name muss mit einem lateinischen Buchstaben beginnen und darf nur Buchstaben, Ziffern und Unterstriche aber keine Leerzeichen enthalten.


Berechtigungsstrategien
------------------------

Da die Zugriffsregeln vererbt werden, müssen Sie als erstes die Root-Liste ``access`` konfigurieren. Wenn Ihre Projekte größtenteils öffentlich sind (oder wenn Sie überhaupt keine Berechtigung benötigen), können Sie ``read`` und ``write`` an "everyone" vergeben::


    ## in der Hauptkonfiguration:

    "access": [
        {
            "type": "allow",
            "mode": ["read", "write"],
            "role": ["everyone"]
        }
    ]



Wenn Sie nun den Zugriff auf ein Objekt, z. B. ein Projekt, einschränken wollen, benötigen Sie zwei Zugriffsregeln. Die erste, um eine bestimmte Rolle zuzulassen. Die zweite, um "alle" zu verwehren::

    ## in the project config:

    "access": [
        {
            "type": "allow",
            "mode": ["read", "write"],
            "role": ["members"]
        },
        {
            "type": "deny",
            "mode": ["read", "write"],
            "role": ["everyone"]
        }
    ]

Wenn die meisten Ihrer Projekte zum Beispiel ein Login erfordern, ist es einfacher, mit einer "deny all"-Regel zu beginnen::

    ## in der Hauptkonfiguration:

    "access": [
        {
            "type": "deny",
            "mode": ["read", "write"],
            "role": ["everyone"]
        }
    ]

Dann erlauben Sie explizit den Zugriff auf bestimmte Objekte ::

    # in der Projektkonfigurationsdatei:

    "access": [
        {
            "type": "allow",
            "mode": ["read", "write"],
            "role": ["members"]
        }
    ]

Normalerweise ist es nicht notwendig, ``execute`` Rechte speziell zu konfigurieren. Wenn Sie sich jedoch dazu entschließen, sollten Sie darauf achten, dass zumindest ``asset`` und ``auth`` Aktionen von jedem ausführbar sind. Andernfalls könnten sich Ihre Benutzer nicht einmal anmelden!


Berechtigungsanbieter
-----------------------

Datei
~~~~~~~

Der Datenanbieter verwendet eine einfache Json-Datei, um Autorisierungsdaten zu speichern. Die Json-Datei ist nur ein Array von "user"-Objekten ::


    [
        {
            "login": "user login",
            "password": "sha512 encoded password",
            "name": "display name for the user",
            "roles": [ "role1", "role2", ...]
        },
        {
            ...
        }
    }

Der Name und der Speicherort der Datei ist Ihnen überlassen. Geben Sie einfach ihren absoluten Pfad in der Konfiguration an. Dann wird automatisch das verschlüsselte Passwort generiert. Verwenden Sie dazu den Befehl ``auth passwd``.


LDAP
~~~~

Der LDAP-Provider kann Benutzer für ein ActiveDirectory oder einen OpenLDAP-Server autorisieren. Sie sollten mindestens eine URL des Servers und ein Regelwerk konfigurieren, um die LDAP-Filter auf die GBD WebSuite Rollennamen anzuwenden. Hier ist eine Beispielkonfiguration unter Verwendung des von `forumsys. com` bereitgestellten LDAP-Testservers.  `<http://www.forumsys.com/tutorials/integration-how-to/ldap/online-ldap-test-server>`_ ::

    {
        "type": "ldap",

        ## the URL format is  "ldap://host:port/baseDN?searchAttribute":

        "url": "ldap://ldap.forumsys.com:389/dc=example,dc=com?uid",

        ## Anmeldeinformationen, um sich an den Server zu binden:

        "bindDN": "cn=read-only-admin,dc=example,dc=com",
        "bindPassword": "password",

        ## Filter auf Rollen abbilden:

        "roles": [

            ## LDAP-Benutzer "euler" hat die GBD WebSuite Rolle "Moderatoren":

            {
                "matches": "(&(cn=euler))",
                "role": "moderators"
            },

            ## alle Mitglieder der LDAP-Gruppe "Mathematiker" haben die GBD WebSuite Rolle "Mitglieder":

            {
                "memberOf": "(&(ou=mathematicians))",
                "role": "members"
            }
        ]
    }
