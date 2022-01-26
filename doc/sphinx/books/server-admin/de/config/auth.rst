Autorisierung
=============

Eine Rolle in der GBD WebSuit wird mit einer einfachen Zeichenkette bezeichnet. Ein Nutzer, der sich mit den Zugangsdaten identifiziert, kann mehrere Rollen besitzen.

Zugangsreglungen
----------------

^REF gws.types.Access

In der Konfiguration können einige Typen von Objekten  verknüpft sein mit Zugangsblock (``access``) Konfigurationen, wie z.B.

- Applikation
- Server Aktion
- Projekt
- Layer
- Druckvorlage

Zusätzlich definieren einige Aktionen interne ``access`` Blöcke für bestimmte Befehle.

Ein ``access`` Block ist eine Liste von Regeln. Jede Regel enthält die Eigenschaften ``role`` (ein Name der Rolle auf die sich die Regel bezieht) und ``type``, welche ist entweder ``allow`` ("erlauben") oder ``deny`` ("verweigern").

Wenn ein Nutzer einen Zugriff auf ein Objekt erfragt, werden alle Regel für dieses Objekt überprüft. Falls eine der Rollen, die der Nutzer besitzt, explizit gefunden wird, ist der Zugriff anhand von ``type`` erlaubt oder verweigert. Ansonsten wird das übergeordnete Objekt geprüft. Falls es kein  übergeordnetes Objekt gibt, d.h. das Root-Objekt wird erreicht, wird der Zugriff verweigert.

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

selektives ``deny``
~~~~~~~~~~~~~~~~~~~

Wenn Ihre Projekte größtenteils öffentlich sind (oder wenn Sie überhaupt keine Berechtigung benötigen), können Sie in der App-Konfig ``allow`` an ``all`` vergeben::

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

selektives ``allow``
~~~~~~~~~~~~~~~~~~~~

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

Diese Aktion ist für die Bearbeitung der Zugangsdaten zuständig und muss freigeschaltet sein wenn Sie Logins verwenden. Wenn Sie die "deny all" Strategie folgen, achten Sie darauf, dass die ``auth`` Aktion für ``all`` zugänglich ist, andernfalls könnten sich Ihre Benutzer nicht einmal anmelden.

Autorisierungsanbieter
----------------------

Die Aufgabe eines Autorisierungsanbieters ist, die Zugangsdaten mit der Quelle zu vergleichen und bei der positiven Antwort, Benutzer Eigenschaften (Vollname, Rollen usw) zurückzugeben.

Die Autorisierungsanbieter sind "verkettet". Wenn ein Login vom ersten Anbieter nicht gefunden wird, wird der zweite konfigurierte Anbieter aufgerufen, und so weiter.  Wird jedoch ein Login gefunden, endet der Autorisierungsvorgang sofort, selbst wenn eine Anmeldung fehlschlagt (z.B. wegen einem falschen Passwort).

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

Der ldap-Provider kann Benutzer gegen ein ActiveDirectory oder einen OpenLDAP-Server autorisieren. Sie sollten mindestens eine URL des Servers und ein Regelwerk konfigurieren, um LDAP-Filter auf GBD WebSuit Rollennamen abzubilden. Hier ist eine Beispielkonfiguration unter Verwendung des von `forumsys.com` bereitgestellten `LDAP-Testservers <http://www.forumsys.com/tutorials/integration-how-to/ldap/online-ldap-test-server>`_ ::

    {
        "type": "ldap",

        ## das Format ist  "ldap://host:port/baseDN?searchAttribute":

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

postgres
~~~~~~~~

^REF gws.ext.auth.provider.postgres.Config

Der postgres-Provider nutzt eine Postrgres Datenbank um die Benutzer zu  autorisieren. Sie sollten zwei SQL ``SELECT`` Abfragen Konfigurieren: eine Autorisierungsabfrage ``authSql`` und eine Ladeabfrage ``uidSql``. Die Autorisierungsabfrage enthält die Platzhalter ``{login}`` und ``{password}`` und muss entweder keine oder exakt eine Zeile zurückgeben, mit folgenden Spalten:

{TABLE}
    *uid* | String oder Nummer | eindeutige User-ID
    *roles* | String | eine kommagetrennte Liste der Rollen, die dieser Nutzer besitzt
    *displayname* | String | Nutzername
    *validuser* | Bool | ``true`` falls dieser Nutzer anmelden kann (und nicht, z.B., gesperrt ist)
    *validpassword* | Bool | ``true`` falls das angegebene Passwort valide ist
{/TABLE}

Die Ergebnisse der Autorisierungsabfrage werden wie folgt ausgewertet:

- keine Zeilen zurückgegeben - zum nächsten Autorisierungsanbieter wechseln
- eine Zeile zurückgegeben und *validuser*/*validpassword* sind ``true`` - Nutzer anmelden
- ansonsten, Zugriff verweigern

Die Ladeabfrage soll einen User-ID Platzhalter ``{uid}`` enthalten und exakt eine Zeile zurückgeben mit den Spalten *uid*, *roles* und *displayname* wie oben beschrieben.

Es gelten keine weiteren Anforderungen an die Struktur der SQL Abfragen.

Konkretes Beispiel: angenommen Sie haben Tabellen ``nutzer`` und ``rollen``, mit diesen Daten: ::

        Tabelle nutzer:
        (Passwörter sind mit ``crypt('...', gen_salt('md5'))`` gehasht)

         id_nutzer | login  |         kennwort_hash              |    vorname     | nachname | aktiv
        -----------+--------+------------------------------------+----------------+----------+-------
                 1 | euler  | $1$slHdN9ik$97YERdtxOM2kBJRhQhMDK0 | Leonhard       | Euler    | t
                 2 | gauss  | $1$4eo7YFb6$.L/WWzqFZgHTRWvWldsBm/ | Carl Friedrich | Gauss    | t
                 3 | newton | $1$y52sZO9i$WTpuxr/KD1sFaxehTKs8z1 | Isaac          | Newton   | f

        Tabelle rollen:
        (stellt eine 1:M Verknüpfung zwischen Nutzern und Rollen dar)

         id_nutzer | rolle_bezeichnung
        -----------+-------------------
                 1 | member
                 1 | moderator
                 1 | expert
                 2 | member
                 2 | expert
                 3 | readonly

In dieser Umgebung kann Ihre Konfiguration wie folgt sein (im ``.cx`` Format): ::

    {
        type "postgres"

        authSql """
            SELECT
                n.id_nutzer AS uid,
                (SELECT string_agg(rolle_bezeichnung, ',') FROM rollen AS r WHERE r.id_nutzer = n.id_nutzer) AS roles,
                n.vorname || ' ' || n.nachname AS displayname,
                n.aktiv AS validuser,
                n.kennwort_hash = crypt({password}, kennwort_hash) AS validpassword
            FROM nutzer AS n
            WHERE n.login = {login}
        """

        uidSql """
            SELECT
                n.id_nutzer AS uid,
                (SELECT string_agg(rolle_bezeichnung, ',') FROM rollen AS r WHERE r.id_nutzer = n.id_nutzer) AS roles,
                n.vorname || ' ' || n.nachname AS displayname
            FROM nutzer AS n
            WHERE n.id_nutzer = {uid}
        """
    }

Autorisierungsmethoden
----------------------

Eine Autorisierungsmethode sorgt dafür, dass die Zugangsdaten vom Nutzer zu einem Anbieter weitergeleitet werden. Alle Methoden unterstützen die Option ``secure`` (Defaultwert ``true``), die angibt, dass diese Methode nur über SSL (sichere Verbindung) verfügbar ist. Falls Sie keine Autorisierungsmethode explizit konfigurieren, wird automatisch nur ``web`` freigeschaltet.

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

Mit dieser Methode werden die Zugangsdaten in HTTP Header mitgesendet. Diese Methode in vor allem für automatische Anmeldungen durch QGIS Plugins und geschützten OWS Dienste gedacht.

Sitzungen
---------

Sitzungen werden in einer Sqlite Tabelle gespeichert, die sich in einem persistenten Ordner innerhalb des ``var`` Ordners befindet. Dies bedeutet, dass die Sitzungen auch nach einem Neustart des Servers nicht unterbrochen werden.

Sie können die Lebenszeit einer Sitzung mit der Option ``sessionLifeTime`` steuern.

^CLIREF auth.sessions

Mit dem Kommandozeilen-Befehl ``gws auth sessions`` können Sie die aktiven Sitzungen auflisten.
