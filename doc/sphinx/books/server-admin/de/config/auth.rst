Autorisierung
=============

Eine Rolle in GWS wird mit einer einfachen Zeichenkette bezeichnet. Ein Nutzer, der sich mit den Zugangsdaten identifiziert, kann mehrere Rollen besitzen.

Zugangsreglungen
----------------

^REF gws.types.Access

In der Konfiguration können einige Typen von Objekten  verknüpft sein mit Zugangsblock (``access``) Konfigurationen, wie z.B.

- main application
- server action
- project
- layer

Zusätzlich definieren einige Aktionen interne ``access`` Blöcke für bestimmte Befehle.

Ein ``access`` Block ist eine Liste von Regeln. Jede Regel enthält die Eigenschaften ``role`` (ein Name der Rolle auf die sich die Regel bezieht) und ``type``, welche ist entweder ``allow`` ("erlauben") oder ``deny`` ("verweigern").

Wenn ein Nutzer X einen Zugriff auf ein Objekt erfragt, werden alle Regel für dieses Objekt überprüft. Falls eine von Rollen die der Nutzer besitzt explizit gefunden wird, ist der Zugriff anhand von ``type`` erlaubt und verweigert. Ansonsten wird das übergeordnete Objekt geprüft. Falls es kein  übergeordnetes Objekt gibt, d.h. das Root-Objekt wird erreicht, ist der Zugriff verweigert.

Vordefinierte Rollen
--------------------

Es gibt einige vordefinierte Rollen, die in GWS eine besondere Bedeutung haben:

{TABLE}
   ``guest`` | nicht eingeloggter Benutzer
   ``user`` | jeder eingeloggter Benutzer
   ``all`` | alle Benutzer, eingeloggt und Gäste. Objekte, auf welche die Rolle ``all`` Zugriff hat sind öffentliche ("public") Objekte
   ``admin`` | Administrator. Benutzer die diese Rolle haben, erhalten automatisch Zugriff auf alle Objekte
{/TABLE}

Andernfalls können Sie beliebige Rollennamen verwenden, aber sie müssen gültige Bezeichnungen sein (d. h. mit einem lateinischen Buchstaben beginnen und nur Buchstaben, Ziffern und Unterstriche enthalten).

Berechtigungsstrategien
-----------------------

Da die Zugriffsregeln vererbt werden, müssen Sie als erstes die Root-Liste ``access`` konfigurieren. Wenn Ihre Projekte größtenteils öffentlich sind (oder wenn Sie überhaupt keine Berechtigung benötigen), können Sie ``allow`` an ``all`` vergeben::

    ## in der App-Konfig:

    "access": [
        {
            "role": "all",
            "type": "allow"
        }
    ]

Wenn Sie nun den Zugriff auf ein Objekt, z. B. ein Projekt, einschränken wollen, benötigen Sie zwei Zugriffsregeln: eine, um eine bestimmte Rolle zuzulassen, und eine, um ``all`` zu verwehren: ::

    ## in der Projektkonfiguration

    "access": [
        {
            "role": "member",
            "type": "allow"
        },
        {
            "role": "all",
            "type": "deny"
        }
    ]

Auf der anderen Seite, wenn die meisten Ihrer Projekte ein Login erfordern, ist es einfacher, mit einer "deny all"-Regel zu beginnen: ::

    ## in der App-Konfig:

    "access": [
        {
            "role": "all",
            "type": "deny"
        }
    ]

und erlauben dann explizit den Zugriff auf bestimmte Objekte: ::

    # in der Projektkonfiguration:

    "access": [
        {
            "role": "member",
            "type": "allow"
        }
    ]

Aktion ``auth``
---------------

^REF gws.ext.action.auth.Config

Diese Aktion ist für die Bearbeitung der Zugangsdaten zuständig und muss freigeschaltet sein wenn Sie Logins verwenden. Wenn Sie die "deny all" Strategie folgen, achten Sie darauf, dass die die ``auth`` Aktion für ``all`` zugänglich ist, andernfalls könnten sich Ihre Benutzer nicht einmal anmelden.

Autorisierungsanbieter
----------------------

Die Aufgabe eines Autorisierungsanbieters ist, die Zugangsdaten mit der Quelle zu vergleichen und bei der positiven Antwort, Benutzer Eigenschaften (Vollname, Rollen usw) zurückzugeben

file
~~~~

^REF gws.ext.auth.provider.file.Config

Der Dateianbieter verwendet eine einfache Json-Datei, um Zugangsdaten zu speichern. Der json ist nur ein Array von "user"-Objekten ::

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

Der Name und der Speicherort der Datei ist Ihnen überlassen, geben Sie einfach ihren absoluten Pfad in der Konfiguration an.

^CLIREF auth.passwd

Um das verschlüsselte Passwort zu generieren, verwenden Sie den Kommandozeilen-Befehl ``gws auth passwd``.

ldap
~~~~

^REF gws.ext.auth.provider.ldap.Config

Der ldap-Provider kann Benutzer gegen ein ActiveDirectory oder einen OpenLDAP-Server autorisieren. Sie sollten mindestens eine URL des Servers und ein Regelwerk konfigurieren, um LDAP-Filter auf GBD WebSuit Rollennamen abzubilden. Hier ist eine Beispielkonfiguration unter Verwendung des von `forumsys.com` bereitgestellten LDAP-Testservers (http://www.forumsys.com/tutorials/integration-how-to/ldap/online-ldap-test-server) ::

    {
        "type": "ldap",

        ## the URL format is  "ldap://host:port/baseDN?searchAttribute":

        "url": "ldap://ldap.forumsys.com:389/dc=example,dc=com?uid",

        ## Anmeldeinformationen, um sich an den Server zu binden:

        "bindDN": "cn=read-only-admin,dc=example,dc=com",
        "bindPassword": "password",

        ## Filter auf Rollen abbilden:

        "users": [

            ## LDAP-Benutzer "euler" hat Rollen "moderator" und "expert":

            {
                "matches": "(&(cn=euler))",
                "roles": ["moderator", "expert"]
            },

            ## alle Mitglieder der LDAP-Gruppe "mathematicians" haben die Rolle "member":

            {
                "memberOf": "mathematicians",
                "roles": ["member"]
            }
        ]
    }

Autorisierungsmethoden
----------------------

Eine Autorisierungsmethode sorgt dafür, dass die Zugangsdaten vom Nutzer zu einem Anbieter weitergeleitet werden. Derzeit sind folgende Methoden implementiert:

web
~~~

^REF gws.ext.auth.method.web.Config

Sendet die Zugangsdaten als eine JSON-Struktur an den Server Endpunkt. Bei der positiven Prüfung setzt der Server ein Sitzungscookie, das bei weiteren Anfragen mitgesendet wird.

Im Browser wird zur Bearbeitung eines Login-Formulars eine Javascript Funktion benötigt, die den Endpunkt mittels AJAX aufruft. Eine beispielhafte Vorlage des Formulars kann wie folgt aussehen: ::

    @if user.is_guest
        ## Login Formular für nicht-eingeloggte Nutzer

        <form onsubmit="gwsLogin()">
            <label>
                Benutzername
                <input type="text" id="gwsUsername" name="username"/>
            </label>
            <label>
                Kennwort
                <input type="password" id="gwsPassword" name="password"/>
            </label>
            <button type="submit">Einloggen</button>
        </form>

    @else
        ## Logout Button für eingeloggte Nutzer

        <button onclick="gwsLogout()">Ausloggen</button>

    @end

Die Definitionen der Funktionen ``gwsLogin`` und ``gwsLogout`` finder Sie unter https://github.com/gbd-consult/gbd-websuite/blob/master/client/src/gws-start.js. Sie können auch eigene Funktionen verwenden.

Siehe auch ^template für Details über die Vorlagen-Sprache.

basic
~~~~~

^REF gws.ext.auth.method.basic.Config

Mit dieser Methode werden die Zugangsdaten in HTTP Header mitgesendet. Diese Methode in vor allem für automatische Anmeldungen durch QGIS Plugins und geschüzten OWS Dienste gedacht.

Sitzungen
---------

Sitzungen werden in einer Sqlite Tabelle gespeichert, die sich in einem persistenten Ordner innerhalb des ``var`` Ordner befindet. Dies bedeutet, dass die Sitzungen auch nach einem Neustart des Servers nicht unterbrochen werden.

Sie können die Lebenszeit einer Sitzung mit der Option ``sessionLifeTime`` steuern.

^CLIREF auth.sessions

Mit dem Kommandozeilen-Befehl ``gws auth sessions`` können Sie die aktiven Sitzungen auflisten.
