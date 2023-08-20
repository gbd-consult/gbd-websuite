import gws.plugin.ows_service.csw.templates.iso_record as rec
import gws.base.ows.server.templatelib as tpl


def main(ARGS):
    return tpl.to_xml(ARGS, rec.record(ARGS, ARGS.record))
