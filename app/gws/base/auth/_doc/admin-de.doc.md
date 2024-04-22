# Auth & Auth:/admin-de/config/auth

**Authentifizierung**: Feststellung der Identität eines Benutzers.

**Authorisierung**: Feststellung ob ein Benutzer eine bestimmte Aktion durchführen 
darf.

## Authentifizierung

Um mit Benutzern, Rollen und Berechtigungen in der GBD WebSuite arbeiten zu 
können muss dieses Feature einmal in der Konfiguration generell aktiviert werden:

{file /data/config.cx}
```javascript
actions+ { type auth }
```

### Methoden

Ein Benutzer der auf eine beliebige Ressource der GBD WebSuite zuzugreifen 
versucht ist zunächst einmal anonym. Um sich gegenüber der Applikation zu 
Identifizieren unterstützt die GBD WebSuite zwei Verfahren über die der Benutzer
eine nur Ihm bekannte Information mitschicken kann, durch die die GBD WebSuite 
ihm eine hinterlegte Identität und die damit verbundenen Berechtigungen zuweisen 
kann.

#### Cookies / web

Der Benutzer gibt auf einer Login Seite seinen Benutzernamen und sein Passwort ein.
Die WebSuite überprüft diese und legt für den Benutzer in einer internen Datenbank 
eine Session an. Die Session hat einen langen, nicht erratbaren Namen welcher in
einem Cookie hinterlegt wird. Der Browser des Benutzers schickt diesen Cookie
bei jeder Anfrage an die GBD WebSuite automatisch mit, wodurch der Benutzer 
identifiziert werden kann.

Solange Benutzer nur über einen Webbrowser mit der GBD WebSuite interagiert ist
dies die beste und einzig nötige Authentifizierungsmethode.

Um diese Methode zu verwenden sind folgende Einträge in der Konfiguration 
vorzunehmen:

{file /data/config.cx}
```javascript
{
    actions+ { type auth }

    auth.methods+ { 
        type web 
        secure false
    }
}
```
%reference_de 'gws.plugin.auth_method.web.core.Config'

Ohne `secure false` zu setzen weigert sich die GBD WebSuite die Authentifizierung
über unverschlüsselte Verbindungen durchzuführen.

Sie können das Login Formular auf einer durch die GBD WebSuite bereitgestellten
Web Seite einbinden:

{file /data/assets/index.cx.html}
```html
...
<script src="/_/webSystemAsset/path/util.js"></script>
@if user.isGuest
    <form onsubmit="gwsLogin(); return false" class="row">
        <input id="gwsUsername" name="username" placeholder="Benutzername">
        <input id="gwsPassword" name="password" type="password" placeholder="Passwort">
        <button id="loginButton">Einloggen</button>
    </form>

    <div id="loginWait"></div>

    <div id="loginErrorMessage">
        Anmeldung fehlgeschlagen!
    </div>
@else
    <button onclick="gwsLogout()">Ausloggen</button>
@end
...
```

#### basic

Bei der Basic Authentifizierung wird bei jeder Anfrage Benutzername und Passwort
mitgeschickt. Möchten Sie mit QGIS auf von der GBD WebSuite bereitgestellte, 
zugriffsgeschützte OWS Dienste zugreifen benötigen Sie diese 
Authentifizierungsmethode:

{file /data/config.cx}
```javascript
{
    actions+ { type auth }

    auth.methods+ { 
        type basic
        secure false
    }
}
```
%reference_de 'gws.plugin.auth_method.basic.core.Config'


### Provider

Nachdem die GBD WebSuite die Zugangsdaten durch eine der oben beschriebenen 
[Methoden](TODO LINK) erhalten hat, muss sie überprüfen ob diese Zugangsdaten 
richtig sind, herausfinden zu welchem Benutzer diese gehören und Informationen 
zu diesem Benutzer herausfinden, wie z.B. der Anzeigename oder die dem Benutzer 
zugewiesenen Rollen.

Diese Informationen kann die GBD WebSuite von einem Authentifizierungsprovider 
beziehen. Es werden folgende Provider unterstützt:

#### file

Mit diesem Authentifizierungsprovider ist es möglich eine Liste von Benutzern in
einer [JSON](https://www.json.org/json-de.html) Datei zu hinterlegen.

Ergänzen Sie die Konfiguration für den Authentifizierungsprovider wie folgt:

{file /data/config.cx}
```javascript
{
    actions+ { type auth }

    auth.methods+ { 
        type web 
        secure false
    }

    auth.providers+ {
        type file
        path "/data/users.json"
    }
}
```

Und hinterlegen Sie eine Liste mit Benutzerkonten in der angegebenen Datei:

{file /data/users.json}
```json
[
    {
        "login": "user_login",
        "password": "sha512_encoded_password",
        "name": "display name for the user",
        "roles": [ "role1", "role2", ...]
    },
    {
        ...
    }
]
```

Um das Passwort nicht im Klartext in der Konfiguration zu hinterlegen wird ein 
[Hashwert](https://de.wikipedia.org/wiki/Passwort#Speichern_von_Passwörtern) 
verwendet. Sie können den in der Datei zu hinterlegenden Hashwert mittels eines 
Kommandozeilenbefehls von der WebSuite erhalten: [`gws auth password`](TODO LINK KOMMANDOZEILENREFERENZ)

#### ldap

Der ldap-Provider kann Benutzer gegen ein ActiveDirectory oder einen 
OpenLDAP-Server authentifizieren. Sie sollten mindestens eine URL des Servers und 
ein Regelwerk konfigurieren, um LDAP-Filter auf GBD WebSuite Rollennamen 
abzubilden. 

Hier ist eine Beispielkonfiguration unter Verwendung des von 
[forumsys.com](https://forumsys.com) bereitgestellten 
[LDAP-Testservers](http://www.forumsys.com/tutorials/integration-how-to/ldap/online-ldap-test-server):

```javascript
{
    type ldap

    url "ldap://ldap.forumsys.com:389/dc=example,dc=com?uid"

    bindDN "cn=read-only-admin,dc=example,dc=com"
    bindPassword "password"

    users [

        // LDAP-Benutzer "euler" hat Rollen "moderator" und "expert":
        {
            matches "(&(cn=euler))"
            roles ["moderator" "expert"]
        }

        // alle Mitglieder der LDAP-Gruppe "mathematicians" haben die Rolle "member":
        {
            memberOf "mathematicians"
            roles ["member"]
        }
    ]
}
```
%reference_de 'gws.plugin.auth_provider.ldap.Config'

#### postgres

TODO

## Authorisierung

Nach der Authentifizierung wird überprüft ob der Benutzer authorisiert ist auf 
die gewünschte Art mit der gewünschten Ressource zu interagieren.

### Rollen

Im Rahmen der Authentifizierung hat der Benutzer eine Liste von Rollen zugewiesen 
bekommen. Es gibt ein paar vom System vorgegebene Rollen die ein Benutzer automatisch
abhängig von seinem Authentifizierungsstatus erhält, und zusätzlich die Rollen die 
laut Authentifizierungsprovider dem Benutzer zugewiesen werden.

GWS Rollen sind Strings (Textfolgen) die mit einem lateinischen Buchstaben 
beginnen, und nur Buchstaben, Ziffern und Unterstriche enthalten dürfen.
Ebenfalls dürfen Sie die Rollen `guest`, `user` und `all` nicht selbst vergeben.


#### vordefinierte Rollen

| Rolle     | Bedeutung |
|-----------|---|
|``guest``  | nicht eingeloggter Benutzer |
|``user``   | jeder eingeloggter Benutzer |
|``all``    | alle Benutzer, eingeloggt und Gäste. Objekte, auf welche die Rolle ``all`` Zugriff hat sind öffentliche ("public") Objekte |
|``admin``  | Diese Rolle kann durch den Authentifizierungsprovider zugewiesen werden, und erhält _immer_ Zugriff auf _alle_ Objekte. |


### Zugriffsregeln

Für einige Objekte in der Konfiguration können Regeln hinterlegt werden, anhand 
derer entschieden wird ob und wie der Benutzer mit diesem interagieren kann.

Das Hinterlegen von diesen Regeln findet auf zwei Arten statt:

#### access

Die meisten Objekte für die eine Zugriffskontrolle möglich ist, haben eine 
Eigenschaft `access`:

{file /data/config.cx}
```javascript
{
    access "allow all"

    project+ {
        access "allow myrole, deny all"
        uid myproject
        title "Mein Projekt"
    }
}
```
%reference_de 'gws.AclStr'

#### permissions

%reference_de 'gws.PermissionsConfig'




Über die Eigenschaft `access` für generellen Zugriff auf ein Objekt
und über permissions.xxx






















TODO Work In Progress SV, alles hier drüber ist soweit fertig.

- Applikation
- Server Aktion
- Projekt
- Layer
- Druckvorlage

Zusätzlich definieren einige Aktionen interne ``access`` Blöcke für bestimmte Befehle.

Ein ``access`` Block ist eine Liste von Regeln. Jede Regel enthält die Eigenschaften ``role`` (ein Name der Rolle auf die sich die Regel bezieht) und ``type``, welche ist entweder ``allow`` ("erlauben") oder ``deny`` ("verweigern").

Wenn ein Nutzer einen Zugriff auf ein Objekt erfragt, werden alle Regel für dieses Objekt überprüft. Falls eine der Rollen, die der Nutzer besitzt, explizit gefunden wird, ist der Zugriff anhand von ``type`` erlaubt oder verweigert. Ansonsten wird das übergeordnete Objekt geprüft. Falls es kein  übergeordnetes Objekt gibt, d.h. das Root-Objekt wird erreicht, wird der Zugriff verweigert.


## Berechtigungsstrategien

### selektives ``deny``

Wenn Ihre Projekte größtenteils öffentlich sind (oder wenn Sie überhaupt keine Berechtigung benötigen), können Sie in der App-Konfig ``allow`` an ``all`` vergeben:

```javascript

"app": {
    "access": [
        {
            "role": "all",
            "type": "allow"
        }
    ]
}
```

Wenn Sie nun den Zugriff auf ein Objekt, z. B. ein Projekt, einschränken wollen, benötigen Sie zwei Zugriffsregeln: eine, um eine bestimmte Rolle zuzulassen, und eine, um ``all`` zu verwehren: ::

```javascript

# project
{ 
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
}
```

### selektives ``allow``

Auf der anderen Seite, wenn die meisten Ihrer Projekte ein Login erfordern, ist es einfacher, mit einer "deny all"-Regel zu beginnen:

```javascript

"app": {
    "access": [
        {
            "role": "all",
            "type": "deny"
        }
    ]
}
```

und erlauben dann explizit den Zugriff auf bestimmte Objekte:

```javascript

#project
{
    "access": [
        {
            "role": "member",
            "type": "allow"
        }
    ]
}
```

## Aktion ``auth``

%reference_de 'gws.base.auth.manager.Config'

Diese Aktion ist für die Bearbeitung der Zugangsdaten zuständig und muss freigeschaltet sein wenn Sie Logins verwenden. Wenn Sie die "deny all" Strategie folgen, achten Sie darauf, dass die ``auth`` Aktion für ``all`` zugänglich ist, andernfalls könnten sich Ihre Benutzer nicht einmal anmelden.

## Autorisierungsanbieter

Die Aufgabe eines Autorisierungsanbieters ist, die Zugangsdaten mit der Quelle zu vergleichen und bei der positiven Antwort, Benutzer Eigenschaften (Vollname, Rollen usw) zurückzugeben.

### file

%reference_de 'gws.plugin.auth_provider.file.Config'

Der Dateianbieter verwendet eine einfache Json-Datei, um Zugangsdaten zu speichern. Der json ist nur ein Array von "user"-Objekten:

```javascript
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
]
```

Der Name und der Speicherort der Datei ist Ihnen überlassen, geben Sie einfach ihren absoluten Pfad in der Konfiguration an.

TODO! ^CLIREF auth.passwd

Um das verschlüsselte Passwort zu generieren, verwenden Sie den Kommandozeilen-Befehl ``gws auth passwd``.

### ldap

TODO! %reference_de 'gws.ext.auth.provider.ldap.Config'

Der ldap-Provider kann Benutzer gegen ein ActiveDirectory oder einen OpenLDAP-Server autorisieren. Sie sollten mindestens eine URL des Servers und ein Regelwerk konfigurieren, um LDAP-Filter auf GBD WebSuit Rollennamen abzubilden. Hier ist eine Beispielkonfiguration unter Verwendung des von `forumsys.com` bereitgestellten [LDAP-Testservers](http://www.forumsys.com/tutorials/integration-how-to/ldap/online-ldap-test-server)

```javascript
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
```

## Autorisierungsmethoden

Eine Autorisierungsmethode sorgt dafür, dass die Zugangsdaten vom Nutzer zu einem Anbieter weitergeleitet werden. Alle Methoden unterstützen die Option ``secure`` (Defaultwert ``true``), die angibt, dass diese Methode nur über SSL (sichere Verbindung) verfügbar ist. Falls Sie keine Autorisierungsmethode explizit konfigurieren, wird automatisch nur ``web`` freigeschaltet.

### web

%reference_de 'gws.plugin.auth_method.web.action.Config'

Sendet die Zugangsdaten als eine JSON-Struktur an den Server Endpunkt. Bei der positiven Prüfung setzt der Server ein Sitzungscookie, das bei weiteren Anfragen mitgesendet wird.

Im Browser wird zur Bearbeitung eines Login-Formulars eine Javascript Funktion benötigt, die den Endpunkt mittels AJAX aufruft. Eine beispielhafte Vorlage des Formulars kann wie folgt aussehen: ::

```html
    @if user.isGuest
        <!-- Login Formular für nicht-eingeloggte Nutzer -->

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
        <!-- Logout Button für eingeloggte Nutzer -->

        <button onclick="gwsLogout()">Ausloggen</button>

    @end
```

Die Definitionen der Funktionen ``gwsLogin`` und ``gwsLogout`` finder Sie unter https://github.com/gbd-consult/gbd-websuite/blob/master/client/src/gws-start.js. Sie können auch eigene Funktionen verwenden.

Siehe auch [Vorlage](/admin-de/config/template) für Details über die Vorlagen-Sprache.

### basic

%reference_de 'gws.plugin.auth_method.basic.Config'

Mit dieser Methode werden die Zugangsdaten in HTTP Header mitgesendet. Diese Methode in vor allem für automatische Anmeldungen durch QGIS Plugins und geschützten OWS Dienste gedacht.

## Sitzungen

Sitzungen werden in einer Sqlite Tabelle gespeichert, die sich in einem persistenten Ordner innerhalb des ``var`` Ordners befindet. Dies bedeutet, dass die Sitzungen auch nach einem Neustart des Servers nicht unterbrochen werden.

Sie können die Lebenszeit einer Sitzung mit der Option ``sessionLifeTime`` steuern.

TODO! ^CLIREF auth.sessions

Mit dem Kommandozeilen-Befehl ``gws auth sessions`` können Sie die aktiven Sitzungen auflisten.
