## Client Localization :/dev-en/client-i18n

Create a javascript file in ``src/lang`` with the name of your language or locale, e.g. ``de.js`` or ``de_DE.js``.

Include this file in ``src/lang/index.js`` e.g.

```
    'de_DE': require('./de_DE')
```

This file must export a single object ``message identifier : translated text``:

```
    module.exports = {
        modUserLoginButton: 'Einloggen',
        modUserLogoutButton: 'Ausloggen',
        printerButton: 'Drucken',

        // etc
    }
```

To list all message identifiers used in the project, run the following command in the ``gws-client`` directory:

```
    find  ./src -regex '.*tsx?$' | xargs grep __
```