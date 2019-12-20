Templating
==========

HTML templates
--------------

An html template is a text/html file, with external variables enclosed in ``{...}``. Additionally, there are basic programming constructs (conditions, loops, functions) that allow changing the template output depending on the variables.

For project, layer and feature templates, the system provides objects ``project``, ``layer`` and ``feature`` with their respective properties that can be used in templating. Here's an example of a feature formatting template ::

    ## this is a comment
    ## format a "city" feature, that has the following attributes: "name", "area", "population"

    @with feature.attributes as atts

        @if atts.population > 100000
            <div class="big-city">{atts.name | html}</div>
        @else
            <div class="small-city">{atts.name | html}</div>
        @end

        <p> <strong>Area:</strong> {atts.area} </p>
        <p> <strong>Population:</strong> {atts.population} </p>

    @end


Refer to the `templating engine documentation <https://github.com/gebrkn/chartreux>`_ for the complete description of all features available.


Configuration templates
-----------------------

Configuration templates (``config.cx``) are similar to html templates and share the same set of programming constructs. An important difference is that variables must be enclosed in *two* braces: ``{{...}}``. A configruation template is supposed to be in the JSON format, additionally, the "shortcut" JSON syntax can be used (`documentation <https://github.com/gebrkn/slon>`_).

Example of a configuration template ::


    ## main application configuration

    @include database-config.cx
    @include server-config.cx

    timeZone "Europe/Berlin"

    ## we have four sites, each of them has its own root dir

    web {
        sites [
            @each [1, 2, 3] as siteIndex
                {
                    host "www{{siteIndex}}.mydomain.com"
                    root.dir "/data/web/{{siteIndex}}"
                    errorPage {
                        type "html"
                        path "/data/templates/error.cx.html"
                    }
                }
            @end
        ]
    }



