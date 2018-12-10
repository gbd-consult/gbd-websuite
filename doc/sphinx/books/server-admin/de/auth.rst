Autorisierung
=============

Die GBD WebSuite Autorisierung ist rollenbasiert, mit steckbaren Berechtigungsanbietern. Wenn sich der Benutzer anmeldet, werden seine Credentials nacheinander an alle konfigurierten Provider weitergegeben, wenn ein Provider die Credentials akzeptiert, gibt er eine Liste der Rollen für diesen Benutzer zurück. 

Wenn der Benutzer ein Objekt anfordert, prüft der Server, ob eine der Rollen des Benutzers über ausreichende Berechtigungen verfügt, um das Objekt zu lesen, zu schreiben (z. B. beim Bearbeiten) oder auszuführen (z. B. eine Serveraktion). Wenn es keine expliziten Berechtigungen auf Objektebene gibt, wird das übergeordnete Objekt geprüft und so weiter. Um diese Prüfungen durchzuführen, liest der Server die ``access`` jedes angeforderten Objekts.  

Zugangsreglungen
----------------


``access`` ist eine Liste von ``AccessRule`` Objekten. Jede ``AccessRule`` enthält

- der ``type`` der Regel - "erlauben" oder "verweigern",
- der ``mode`` - "lesen", "schreiben", "ausführen" oder eine Kombination davon,
- die Liste der ``role`` Namen, auf die sich die Regel bezieht


Mit Hilfe der ``access``-Regeln kann der Berechtigungsprüfungsalgorithmus formal wie folgt beschrieben werden::


    ## Der Benutzer U fordert einen Berechtigungsmodus P (z. B. "lesen") für ein Objekt O

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

        ## An diesem Punkt haben wir das Wurzelobjekt erreicht.
        ## und haben immer noch keine passende Regel gefunden.
        ## Verwenden Sie die Standardregel "Alle Anfragen ablehnen".

        return Access Denied

    end loop

Rollen
----------

Es gibt einige vordefinierte Rollen, die in GWS eine besondere Bedeutung haben:

TABLE
   *guest* ~ Nicht eingeloggter Benutzer
   *user* ~ Jeder eingeloggter Benutzer
   *everyone* ~ Alle Benutzer, eingeloggt und Gäste
   *admin* ~ Administrator. Benutzer die diese Rolle haben, erhalten automatisch Zugriff auf alle Ressourcen
/TABLE

Andernfalls können Sie beliebige Rollennamen verwenden, aber sie müssen gültige Bezeichnungen sein (d. h. mit einem lateinischen Buchstaben beginnen und nur Buchstaben, Ziffern und Unterstriche enthalten). 


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



Wenn Sie nun den Zugriff auf ein Objekt, z. B. ein Projekt, einschränken wollen, benötigen Sie zwei Zugriffsregeln: eine, um eine bestimmte Rolle zuzulassen, und eine, um "alle" zu verwehren:: 

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

Auf der anderen Seite, wenn die meisten Ihrer Projekte ein Login erfordern, ist es einfacher, mit einer "deny all"-Regel zu beginnen:: 

    ## in der Hauptkonfiguration: 

    "access": [
        {
            "type": "deny",
            "mode": ["read", "write"],
            "role": ["everyone"]
        }
    ]

und erlauben dann explizit den Zugriff auf bestimmte Objekte ::

    # in der Projektkonfigurationsdatei:

    "access": [
        {
            "type": "allow",
            "mode": ["read", "write"],
            "role": ["members"]
        }
    ]

Normalerweise ist es nicht notwendig, ``execute`` Rechte speziell zu konfigurieren, aber wenn Sie sich dazu entschließen, sollten Sie darauf achten, dass zumindest ``asset`` und ``auth`` Aktionen von jedem ausführbar sind, andernfalls könnten sich Ihre Benutzer nicht einmal anmelden!


Berechtigungsanbieter
-----------------------

Datei
~~~~~~~

Der Dateianbieter verwendet eine einfache Json-Datei, um Autorisierungsdaten zu speichern. Der json ist nur ein Array von "user"-Objekten ::


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

Der Name und der Speicherort der Datei ist Ihnen überlassen, geben Sie einfach ihren absoluten Pfad in der Konfiguration an. Um das verschlüsselte Passwort zu generieren, verwenden Sie den Befehl ``auth passwd``.


Ldap
~~~~

Der ldap-Provider kann Benutzer gegen ein ActiveDirectory oder einen OpenLDAP-Server autorisieren. Sie sollten mindestens eine URL des Servers und ein Regelwerk konfigurieren, um LDAP-Filter auf GBD WebSuit Rollennamen abzubilden. Hier ist eine Beispielkonfiguration unter Verwendung des von `forumsys. com bereitgestellten LDAP-Testservers.  <http://www.forumsys.com/tutorials/integration-how-to/ldap/online-ldap-test-server>`_ ::

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

