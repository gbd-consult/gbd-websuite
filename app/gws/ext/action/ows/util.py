import gws.tools.xml3
import gws.types as t
import gws.common.project
import gws.web.auth


def xml_response(tag, content=None, attributes=None):
    return t.HttpResponse({
        'mimeType': 'text/xml',
        'content': gws.tools.xml3.string(tag, content, attributes)
    })


class RequestParams(t.Data):
    action: t.ObjectInterface
    project: gws.common.project.Object
    req: gws.web.auth.AuthRequest
    ps: dict
