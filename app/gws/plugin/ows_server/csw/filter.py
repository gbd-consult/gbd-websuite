import gws
import gws.lib.crs
import gws.lib.extent
import gws.lib.bounds


class Filter:
    def __init__(self, index):
        self.index = index

    def apply(self, flt, recs):
        fn = getattr(self, '_' + flt.name.lower(), None)
        if not fn:
            return []
        return fn(flt, recs)

    def _and(self, flt, recs):
        for el in flt.all():
            recs = self.apply(el, recs)
        return recs

    def _or(self, flt, recs):
        ns = set()
        for el in flt.all():
            ns.update(r.index for r in self.apply(el, recs))
        return [r for r in recs if r.index in ns]

    def _bbox(self, flt, recs):
        b = gws.lib.bounds.from_gml_envelope_element(flt.first('Envelope'))
        if not b:
            return []
        b = gws.lib.bounds.transformed_to(b, gws.lib.crs.get4326())
        return [
            r for r in recs
            if not r.get('wgsExtent') or gws.lib.extent.intersect(r.wgsExtent, b.extent)
        ]

    def _propertyislike(self, flt, recs):
        # @TODO wildcards

        try:
            prop = (flt.first('propertyname').text or 'csw:AnyText').lower()
            val = flt.first('literal').text
        except TypeError:
            return []

        if prop == 'csw:anytext':
            ns = set(
                idx for f, s, lows, idx in self.index
                if val in s
            )
        else:
            ns = set(
                idx for p, s, lows, idx in self.index
                if val in s and p == prop
            )

        return [r for r in recs if r.index in ns]
