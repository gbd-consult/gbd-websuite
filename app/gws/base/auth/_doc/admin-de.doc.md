# Auth & Auth:/admin-de/themen/auth

**Authentifizierung**: Feststellung der Identität eines Benutzers.

**Authorisierung**: Feststellung ob ein Benutzer eine bestimmte Aktion durchführen 
darf.

## Authentifizierung

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

```javascript title="/data/config.cx"
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

Die [Auth Action](TODO LINK) aktiviert die Login und Logout Endpunkt die vom 
dem Formular angesprochen werden.

Sie können das Login Formular auf einer durch die GBD WebSuite bereitgestellten
Web Seite einbinden:

```html title="/data/assets/index.cx.html"
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

```javascript title="/data/config.cx"
{
    auth.methods+ { 
        type basic
        secure false
    }
}
```
%reference_de 'gws.plugin.auth_method.basic.Config'


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

```javascript title="/data/config.cx"
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

```json title="/data/users.json"
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

%warn
Die Eigenschaft `access` sollte nicht mehr verwendet werden. Verwenden Sie
stattdessen `permissions.read`. Sind beide Eigenschaften gesetzt wird 
`permissions.read` immer bevorzugt. `access` wird mit einem zukünftigen
Release entfernt.
%end

Die meisten Objekte für die eine Zugriffskontrolle möglich ist, haben eine 
Eigenschaft `access`:

```javascript title="/data/config.cx"
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
%reference_de 'gws.AclStr'


`permissions` definiert mit den Eigenschaften `read`, `write`, `update` und 
`delete` die Berechtigungen von Rollen auf Objekte.

Zusätzlich gibt es `permissions.edit` welches ermöglicht `write`, `update` und 
`delete` in einem Befehl auf den gleichen Wert zu setzen.

#### ACL

ACL Strings sind Zeichenketten die für die Eigenschaften des `permissions` 
Blocks hinterlegbar sind.

Sie enthalten eine kommaseparierte Liste von Zugriffsdirektiven bestehend aus
`allow <rolle>` oder `deny <rolle>`.

Die Direktiven werden sequentiell überprüft bis eine dem zugreifenden User
zugewiesene Rolle gefunden wird.

Dem Nutzer wird der Zugriff daraufhin entweder gewährt (`allow`) oder 
verwährt (`deny`), und alle folgenden Direktiven werden ignoriert.

Ist für ein Objekt keine Zugriffsregelung hinterlegt wird das übergeordnete 
Objekt überprüft und die dort hinterlegten Regeln angewandt.
Falls es kein  übergeordnetes Objekt gibt, d.h. das Root-Objekt wird erreicht, 
wird der Zugriff verweigert.

In diesem Beispiel wird allen Zugreifenden Anwendern das Lesen aller 
untergeordneten Objekte erlaubt, und lediglich Usern mit der Rolle `schreibrolle`
das editieren gestattet:

```javascript
{
    permissions.read "allow all"
    permissions.edit "allow schreibrolle, deny all"
}
```

### Berechtigungsstrategien

#### selektives ``deny``

Wenn Ihre Projekte größtenteils öffentlich sind (oder wenn Sie überhaupt keine Berechtigung benötigen), können Sie in der App-Konfig ``allow`` an ``all`` vergeben:

```javascript title="/data/config.cx"
{
    permissions.read "allow all"
    ...
}
```

Wenn Sie nun den Zugriff auf ein Objekt, z. B. ein Projekt, einschränken wollen, benötigen Sie zwei Zugriffsregeln: eine, um eine bestimmte Rolle zuzulassen, und eine, um ``all`` zu verwehren: ::

```javascript title="/data/config/projects/myproject.cx"
{ 
    uid myproject
    title "Mein Projekt"
    permissions.read "allow leserolle, deny all"
    ...
}
```

### selektives ``allow``

Auf der anderen Seite, wenn die meisten Ihrer Projekte ein Login erfordern, ist es einfacher, mit einer "deny all"-Regel zu beginnen:

```javascript title="/data/config.cx"
{
    permissions.read "deny all"
    permissions.edit "deny all"
    ...
}
```

und dann explizit den Zugriff auf bestimmte Objekte zu erlauben:

```javascript title="/data/config/projects/myproject.cx"
{
    uid myproject
    title "Mein Projekt"
    permissions.read "allow leserolle"
    permissions.edit "allow schreibrolle"
    ...
}
```

%info
Bestimmte Aktionen müssen unter umständen für alle Anwender verfügbar sein um 
sich einloggen zu können, und bei dieser Strategie explizit mit 
`permissions.read "allow all"` versehen werden. Für den Web-Formular basierten 
Login wären dies die Aktion `auth`, sowie wahrscheinlich die Aktion `web` um das
Login-Formular als Webseite darstellen zu können.
%end

### Sitzungen

TODO dieser Abschnitt ist noch alt und muss überprüft werden.

Sitzungen werden in einer Sqlite Tabelle gespeichert, die sich in einem persistenten Ordner innerhalb des ``var`` Ordners befindet. Dies bedeutet, dass die Sitzungen auch nach einem Neustart des Servers nicht unterbrochen werden.

Sie können die Lebenszeit einer Sitzung mit der Option ``sessionLifeTime`` steuern.

TODO! ^CLIREF auth.sessions

Mit dem Kommandozeilen-Befehl ``gws auth sessions`` können Sie die aktiven Sitzungen auflisten.
