# Documentation :/dev-en/documentation

## Localizing the configuration reference 

The [configuration reference](/admin-de/reference) is generated automatically from the source code.

To make the reference use localized strings, edit the file `app/gws/spec/strings.de.ini` in the `gbd-websuite` directory. The structure is just a list of `key=value` pairs, where a key is a full name of a config option and value is your translation. The full name is normally a page title + the value in the "Property" column.  

Run the spec generator in the verbose mode to see which translations are missing:

```
make.sh spec -v
```



## :dog
