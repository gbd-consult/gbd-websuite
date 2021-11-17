import gws.plugin.ows_service.csw.templates.iso_record as rec
import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    return tpl.to_xml(ARGS, rec.record(ARGS, ARGS.record))
