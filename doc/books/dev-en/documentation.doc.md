# Documentation :/dev-en/documentation

## Localizing the configuration reference 

The [configuration reference](/admin-de/reference) is generated automatically from the source code.

English descriptions of configuration options are taken directly from sources, for other languages we use a simple key-value translation system. Each object and each property has a unique key, which is a combination of the object path and the property name (add `?dev=1` to the URL to see the keys). To change the translated text, locate `strings.ini` in a `_doc` subdirectory of the corresponding module. For example, to change this key

```
gws.plugin.auth_method.basic.Config.realm
```

you need to edit this file:

```
gbd-websuite/app/gws/plugin/auth_method/_doc/strings.ini
```

In practice, it's often easier to find the key using global search in the project directory.

Once the text is changed, you need to rebuild the docs with `make.sh doc`. Doc development server will regenerate the docs automatically when you save the file.

## Building custom documentation

You can use the GWS documentation generator to build custom documentation for your projects and plugins. To do that, write your docs in the `doc.md` format, as described in [](/dev-en/documentation/dog), create a `json` file with your options (see [](/dev-en/documentation/dog/options)) and pass it to `make.sh doc`:

```
make.sh doc -opt /path/to/my-options.json
```

You can use any `dog` option in `my-options.json`, but the only mandatory one is `docRoots`, which tells the generator where your documentation is located. 

If you want to create a separate documentation website for your project only, specify your root directory in the options file and ensure your documentation has the "root section", for example:

`my-options.json:`
```
{ "docRoots": ["/path/to/my/docs"] }
```

`/path/to/my/docs/index.doc.md:`
```
# My Documentation :/

Welcome!
```

If you want to integrate your documentation in GWS core docs, specify both your directory and the GWS directory in the options file and mark your start section as `:/extra/<name>`. For example:

`my-options.json:`
```
{ "docRoots": ["/path/to/my/docs", "/path/to/gbd-websuite"] }
```

`/path/to/my/docs/index.doc.md:`
```
# My Documentation :/extra/my-project

Welcome!
```

Your documentation will appear on the top level of the GWS docs, along with other "books".

## :dog
