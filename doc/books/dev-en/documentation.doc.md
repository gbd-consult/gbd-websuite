# Documentation :/dev-en/documentation

## Localizing the configuration reference 

The [configuration reference](/admin-de/reference) is generated automatically from the source code.

To make the reference use localized strings, edit the file `app/gws/spec/strings.de.ini` in the `gbd-websuite` directory. The structure is just a list of `key=value` pairs, where a key is a full name of a config option and value is your translation. The full name is normally a page title + the value in the "Property" column.  

Run the spec generator in the verbose mode to see which translations are missing:

```
make.sh spec -v
```

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
