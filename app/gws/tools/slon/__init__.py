"""This is the SLON parser"""

from .decode import Decoder, DecodeError


def loads(s, **opts):
    return Decoder(opts).decode(s)


def load(fp, **opts):
    return Decoder(opts).decode(fp.read())
