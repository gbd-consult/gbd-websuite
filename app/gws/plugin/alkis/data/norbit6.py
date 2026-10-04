"""Read data from Norbit plugin tables (GeoInfoDok 6)."""

import gws
import gws.base.database
import gws.lib.datetimex
import gws.lib.sa as sa
import gws.plugin.postgres.provider

from .geo_info_dok import gid6 as gid
from . import types as dt


class Object(dt.Reader):
    """Reader for ALKIS source tables written by the norBIT ALKIS import (GeoInfoDok 6).

    There is one table per object type, named after the class in lower case,
    with one column per attribute. Struct attributes are stored in columns
    prefixed with the attribute name, lists are stored as arrays. The reader
    converts each row into an instance of the ``gid6`` class, using a reader
    method per attribute, chosen by the attribute type.
    """

    STD_READERS = {
        'Area': 'as_float',
        'Boolean': 'as_bool',
        'CharacterString': 'as_str',
        'Integer': 'as_int',
        'DateTime': 'as_date',
        'Date': 'as_date',

        'AX_Lagebezeichnung': 'as_ax_lagebezeichnung',
        'AX_Buchung_HistorischesFlurstueck': 'as_ax_buchung_historischesflurstueck',
    }
    """Reader method names for basic and special attribute types."""

    def __init__(self, provider: gws.plugin.postgres.provider.Object, schema='public'):
        """Create a reader and set up attribute readers for all GeoInfoDok classes.

        Args:
            provider: Database provider.
            schema: Schema with the source tables.
        """

        self.db = provider
        self.schema = schema

        self.readers = {}

        for meta in gid.METADATA.values():
            if meta['kind'] not in {'struct', 'object'}:
                continue
            d = {}
            for sup in meta['supers']:
                d.update(self.readers.get(sup, {}))
            for attr in meta['attributes']:
                d[attr['name']] = [
                    attr['name'],
                    attr['name'].lower(),
                    getattr(gid, attr['type'], None),
                    self.get_reader(attr)
                ]

            self.readers[meta['name']] = d

    def get_reader(self, attr):
        """Return the reader method for an attribute.

        Args:
            attr: Attribute metadata from the ``gid6`` schema.

        Returns:
            A bound reader method. Unknown types are read as strings.
        """

        typ = attr['type']
        is_list = attr['list']

        std = self.STD_READERS.get(typ)
        if std:
            return getattr(self, std + '_list' if is_list else std)

        meta = gid.METADATA.get(typ)
        if meta:
            if meta['kind'] == 'object':
                return self.as_ref_list if is_list else self.as_ref
            if meta['kind'] == 'struct':
                return self.as_struct_list if is_list else self.as_struct
            if meta['kind'] == 'enum':
                return self.as_enum_list if is_list else self.as_enum

        return self.as_str_list if is_list else self.as_str

    ##

    def count(self, cls, table_name=None):
        # NB not using db.count to avoid schema introspection
        sql = f"SELECT COUNT(*) FROM {self.schema}.{table_name or cls.__name__.lower()}"
        with self.db.connect() as conn:
            try:
                rs = list(conn.execute(sa.text(sql)))
                return rs[0][0]
            except sa.Error:
                conn.rollback()
                return 0

    def read_all(self, cls, table_name=None, uids=None):
        sql = f"SELECT * FROM {self.schema}.{table_name or cls.__name__.lower()}"
        if uids:
            sql += ' WHERE gml_id IN (:uids)'
            sql = sa.text(sql).bindparams(uids=uids)
        else:
            sql = sa.text(sql)

        with self.db.connect() as conn:
            for row in conn.execute(sql, execution_options={'stream_results': True}):
                r = gws.u.to_dict(row)
                o = self.as_struct(cls, '', r)
                o.identifikator = r.get('gml_id')
                o.geom = r.get('wkb_geometry', '')
                yield o

    ##

    def as_struct(self, cls, prop, r):
        """Read a struct attribute.

        A column ``<prop>_<attribute>`` is used if present, otherwise the plain
        attribute column.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name prefix of the struct attributes.
            r: Table row as a dict.

        Returns:
            An instance of ``cls``.
        """

        d = {}

        for attr_name, attr_low_name, attr_cls, fn in self.readers[cls.__name__].values():
            comp_key = prop.lower() + '_' + attr_low_name
            if comp_key in r:
                d[attr_name] = fn(attr_cls, comp_key, r)
            else:
                d[attr_name] = fn(attr_cls, attr_low_name, r)

        o = cls()
        vars(o).update(d)
        return o

    def as_struct_list(self, cls, prop, r):
        """Read a list of structs from parallel array columns.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name prefix of the struct attributes.
            r: Table row as a dict.

        Returns:
            A list of ``cls`` instances.
        """

        d = {}

        for attr_name, attr_low_name, attr_cls, fn in self.readers[cls.__name__].values():
            comp_key = prop.lower() + '_' + attr_low_name
            if comp_key in r:
                a = _array(r[comp_key])
            else:
                a = _array(r.get(attr_low_name))
            if a:
                d[attr_low_name] = a

        objs = []

        for vals in zip(*d.values()):
            r2 = dict(zip(d, vals))
            o = self.as_struct(cls, '', r2)
            objs.append(o)

        return objs

    def as_ref(self, cls, prop, r):
        """Read a reference to another object.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            The identifier of the referenced object.
        """

        return r.get(prop)

    def as_ref_list(self, cls, prop, r):
        """Read a list of references to other objects.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of object identifiers.
        """

        return _array(r.get(prop))

    def as_enum(self, cls, prop, r):
        """Read a code list value.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            An ``EnumPair``, or ``None`` if the value is not in the code list.
        """

        v = str(r.get(prop))
        if v in cls.VALUES:
            return dt.EnumPair(v, cls.VALUES[v])

    def as_enum_list(self, cls, prop, r):
        """Read a list of code list values.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of ``EnumPair`` objects. Values not in the code list are skipped.
        """

        ls = []

        for v in _array(r.get(prop)):
            v = str(v)
            if v in cls.VALUES:
                ls.append(dt.EnumPair(v, cls.VALUES[v]))

        return ls

    ##

    def as_ax_lagebezeichnung(self, cls, prop, r):
        """Read an ``AX_Lagebezeichnung`` attribute.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            The unencoded location string if present, otherwise an ``AX_VerschluesselteLagebezeichnung``.
        """

        if r.get('unverschluesselt'):
            return r.get('unverschluesselt')
        return self.as_struct(gid.AX_VerschluesselteLagebezeichnung, prop, r)

    def as_ax_buchung_historischesflurstueck_list(self, cls, prop, r):
        """Read a list of ``AX_Buchung_HistorischesFlurstueck`` structs.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of ``AX_Buchung_HistorischesFlurstueck`` instances with their ``buchungsblattbezirk`` set.
        """

        # this one is stored as
        #     "blattart": [xxxx],
        #     "buchungsart": ["xxxx"],
        #     ....etc
        #     "buchungsblattbezirk_bezirk": ["xxxx"],
        #     "buchungsblattbezirk_land": ["xx"],
        #     "buchungsblattkennzeichen": ["xxxx"],
        #     "buchungsblattnummermitbuchstabenerweiterung": ["xxxx"],

        bbs = self.as_struct_list(gid.AX_Buchung_HistorischesFlurstueck, '', r)
        bzs = self.as_struct_list(gid.AX_Buchungsblattbezirk_Schluessel, 'buchungsblattbezirk', r)
        objs = []
        for bb, bz in zip(bbs, bzs):
            bb.buchungsblattbezirk = bz
            objs.append(bb)
        return objs

    ##

    def as_str(self, cls, prop, r):
        """Read a string.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A stripped string, or ``None`` if the value is empty.
        """

        return _str(r.get(prop))

    def as_str_list(self, cls, prop, r):
        """Read a list of strings.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of stripped strings, empty values become ``None``.
        """

        return [_str(v) for v in _array(r.get(prop))]

    def as_bool(self, cls, prop, r):
        """Read a boolean.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A boolean, or ``None`` if the value is missing.
        """

        return _bool(r.get(prop))

    def as_bool_list(self, cls, prop, r):
        """Read a list of booleans.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of booleans.
        """

        return [_bool(v) for v in _array(r.get(prop))]

    def as_int(self, cls, prop, r):
        """Read an integer.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            An integer, or ``None`` if the value is missing.
        """

        return _int(r.get(prop))

    def as_int_list(self, cls, prop, r):
        """Read a list of integers.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of integers.
        """

        return [_int(v) for v in _array(r.get(prop))]

    def as_float(self, cls, prop, r):
        """Read a float.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A float, or ``None`` if the value is missing.
        """

        return _float(r.get(prop))

    def as_float_list(self, cls, prop, r):
        """Read a list of floats.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of floats.
        """

        return [_float(v) for v in _array(r.get(prop))]

    def as_date(self, cls, prop, r):
        """Read a date or a date and time.

        ISO strings are parsed, other values are returned as is.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A date or datetime, or ``None`` if the value is empty.
        """

        return _datetime(r.get(prop))

    def as_date_list(self, cls, prop, r):
        """Read a list of dates.

        Args:
            cls: GeoInfoDok class of the attribute.
            prop: Column name.
            r: Table row as a dict.

        Returns:
            A list of dates or datetimes.
        """

        return [_datetime(v) for v in _array(r.get(prop))]


def _str(v):
    """Convert to a stripped string, ``None`` if empty."""

    if v is not None:
        v = str(v).strip()
        if v:
            return v


def _bool(v):
    return bool(v) if v is not None else None


def _int(v):
    return int(v) if v is not None else None


def _float(v):
    return float(v) if v is not None else None


def _datetime(v):
    """Convert an ISO string to a datetime, pass other values through."""

    if not v:
        return None
    if isinstance(v, str):
        return gws.lib.datetimex.from_iso_string(v)
    return v


def _array(val):
    """Wrap a scalar into a list, ``None`` becomes an empty list."""

    if isinstance(val, list):
        return val
    if val is None:
        return []
    return [val]
