"""Schema generator.

Generate python APIs and object databases from GeoInfoDok sources.

For version 6 use RR cat files Basisschema.cat and Fachschema.cat
For version 7 use the QEA (sqlite) file AAA-7.1.2.qea

Usage::

    generator.py 6 /path/to/Basisschema.cat /path/to/Fachschema.cat
    generator.py 7 /path/to/AAA-7.1.2.qea

"""

import re
import os
import json
import textwrap
import sys
import html
import sqlalchemy as sa


def main(version, *paths):
    if version == '6':
        nodes = Parser6().parse(paths)

    elif version == '7':
        nodes = Parser7().parse(paths)

    else:
        raise ValueError('invalid version')

    # dumps(f'{CDIR}/db{version}.json', db)

    py = PythonGenerator(nodes, version).build()

    with open(f'{CDIR}/gid{version}.py', 'w') as fp:
        fp.write(py)


##

CDIR = os.path.dirname(__file__)
TAB = ' ' * 4
TAB2 = TAB * 2
Q3 = '"""'
WRAP_WIDTH = 110

CATEGORY_ROOTS = {
    'AFIS-ALKIS-ATKIS Fachschema': 'fs',
    'AAA Basisschema': 'bs',
    'AAA_Objektartenkatalog': 'ak',
}

T_CLASS = 'class'
T_CATEGORY = 'category'
T_ENUM = 'enum'
T_UNION = 'union'

PY_HEAD = '''\
"""GeoInfoDok <VERSION> schema.

(c) 2023 Arbeitsgemeinschaft der Vermessungsverwaltungen der Länder der Bundesrepublik Deutschland

https://www.adv-online.de/GeoInfoDok/

This code is automatically generated from .CAT/.QEA source files.
"""

from typing import Any, Literal, Optional, TypeAlias, Union
from datetime import date, datetime


# gws:nospec

class Object:
    pass


class Category:
    pass


class Enumeration:
    pass


def object__getattr__(self, item):
    if item.startswith('_'):
        raise AttributeError()
    return None


setattr(Object, '__getattr__', object__getattr__)
setattr(Category, '__getattr__', object__getattr__)
setattr(Enumeration, '__getattr__', object__getattr__)

'''

STD_TYPES = {
    'Angle': 'float',
    'Area': 'float',
    'Boolean': 'bool',
    'CharacterString': 'str',
    'Date': 'date',
    'DateTime': 'datetime',
    'Distance': 'float',
    'GenericName': 'str',
    'Integer': 'int',
    'Length': 'int',
    'LocalName': 'str',
    'Measure': 'str',
    'Query': 'str',
    'Real': 'float',
    'SC_CRS': 'str',
    'URI': 'str',
    'URL': 'str',
    'Volume': 'float',
}


##

class Node:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def __getattr__(self, item):
        return None


##

class Parser:
    nodes: list[Node] = []

    def finalize(self):
        for node in self.nodes:
            self.make_key(node)

        self.filter_category_roots()
        self.resolve_supers()

        for node in self.nodes:
            self.check_flag(node, 'is_aa', 'AA_Objekt')
            self.check_flag(node, 'is_reo', 'AA_REO')

        return [node for node in self.nodes if node.T]

    def make_key(self, node):
        if node.key:
            return node.key

        parent_key = ''
        parent = popattr(node, 'pParent')
        if parent:
            parent_key = self.make_key(parent)

        node.key = parent_key + '/' + node.name.lower()
        return node.key

    def filter_category_roots(self):
        new_nodes = []
        roots = {'/' + to_name(k).lower(): '/' + v for k, v in CATEGORY_ROOTS.items()}

        for node in self.nodes:
            for k, v in roots.items():
                if k in node.key:
                    _, _, rest = node.key.partition(k)
                    node.key = v + rest
                    new_nodes.append(node)

        self.nodes = new_nodes

    def resolve_supers(self):
        for node in self.nodes:
            for sup_name in popattr(node, 'pSuperNames', []):
                sup = self.find_node(sup_name)
                if sup and sup != node:
                    node.supers.append(sup)

    def check_flag(self, node, prop, root):
        a = getattr(node, prop)
        if a is not None:
            return a
        if node.name == root:
            v = True
        elif node.supers:
            v = any(self.check_flag(super_node, prop, root) for super_node in node.supers)
        else:
            v = False
        setattr(node, prop, v)
        return v

    def find_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node

    def get_doc(self, rec):
        s = rec.get('documentation') or rec.get('Note') or rec.get('Notes') or ''
        s = s.strip()
        s = html.unescape(s)
        # remove the [X] prefix, as in
        # "[E] 'Person' ist eine natürliche...
        if s.startswith('['):
            return s.partition(']')[-1].strip()
        return s

    def get_hname(self, node):
        # sometimes, the first quoted word in the name of the object, as in
        # sonstigeEigenschaft: "'Sonstige Eigenschaft' sind Informationen zum Grenzpunkt...
        #
        # however, this should not be extracted
        # weistAuf: "'Flurstück' weist auf 'Lagebezeichnung mit Hausnummer'...

        if not node.doc:
            return

        cmp_name = to_name(node.name)
        cmp_name = re.sub(r'[A-Z]+_(.+)', r'\1', cmp_name)
        cmp_name = cmp_name.replace('_', '').lower()

        patterns = [
            r"^\'(.+?)\'",
            r"^\"(.+?)\"",
            r"^(.+?)\.$",
            r"^(\w+)",
        ]

        for pat in patterns:
            m = re.match(pat, node.doc)
            if m:
                s = m.group(1)
                c = to_name(s).replace('_', '').lower()
                if c == cmp_name:
                    return s

        if node.name == 'funktion':
            # fix a spelling mistake in some docstrings
            return 'Funktion'

    def add_enum_value(self, node, k, v):
        if k is None:
            k = len(node.values) + 1
        node.values[k] = v

    def set_type_from_record(self, node, rec):
        self.set_type_from_string(node, rec.get('type', '') or rec.get('Type', ''))

    def set_type_from_string(self, node, s):
        m = re.match(r'(Sequence|Set)<(.+?)>', s)
        if m:
            node.type = m.group(2)
            node.list = True
        else:
            node.type = s

    def set_cardinality_from_string(self, node, s=None):
        if not s:
            return
        elif s == '0..1':
            node.optional = True
        elif '..' in s:
            node.list = True

    def set_cardinality_from_record(self, node, rec):
        lb = str(rec['LowerBound'])
        ub = str(rec['UpperBound'])
        if lb == '0' and ub == '1':
            node.optional = True
        elif lb != ub:
            node.list = True


class Parser6(Parser):
    def parse(self, paths):
        for path in paths:
            cat = CatParser().parse(path)
            self.parse_object(cat[1], None)
            self.parse_associations(cat[1])
        return self.finalize()

    def parse_object(self, rec, parent):
        node = Node(name=rec['NAME'], pParent=parent, doc=self.get_doc(rec))
        self.nodes.append(node)

        # e.g. zugriffsartProduktkennungBenutzung [0..*]
        m = re.match(r'^(\S+)\s*\[(.+?)]$', node.name)
        if m:
            node.name = m.group(1)
            self.set_cardinality_from_string(node, m.group(2))

        node.hname = self.get_hname(node)
        node.name = to_name(node.name)

        for a in rec.get('attributes', []):
            if a['name'] == 'Kennung':
                node.uid = a['value']

        stereo = rec.get('stereotype', '').lower()

        if rec['TYPE'] == 'Class_Category':
            node.T = T_CATEGORY
            for r2 in rec.get('logical_models', []):
                self.parse_object(r2, node)

        elif rec['TYPE'] == 'Class' and stereo in {'codelist', 'enumeration'}:
            node.T = T_ENUM
            node.values = {}
            for r2 in rec.get('class_attributes', []):
                self.add_enum_value(node, r2.get('initv'), r2['NAME'])

        elif rec['TYPE'] == 'Class' and stereo == 'union':
            node.T = T_UNION
            node.attributes = [
                self.parse_object(r2, node)
                for r2 in rec.get('class_attributes', [])
            ]

        elif rec['TYPE'] == 'Class':
            node.T = T_CLASS
            node.attributes = [
                self.parse_object(r2, node)
                for r2 in rec.get('class_attributes', [])
            ]
            node.supers = []
            node.pSuperNames = [
                r2['supplier'].split(':')[-1]
                for r2 in rec.get('superclasses', [])
            ]

        elif rec['TYPE'] == 'ClassAttribute':
            self.set_type_from_record(node, rec)

        elif rec['TYPE'] == 'Role':
            self.set_type_from_string(node, rec['supplier'].split(':')[-1])
            self.set_cardinality_from_string(node, rec.get('client_cardinality'))

        return node

    def parse_associations(self, rec):
        if rec['TYPE'] != 'Association':
            for o2 in rec.get('logical_models', []):
                self.parse_associations(o2)
            return

        # This is what an association looks like in a parsed .cat:
        #
        # {
        #     "NAME": "$UNNAMED$38",
        #     "TYPE": "Association",
        #     "quid": "40FED632018C",
        #     "roles": [
        #         {
        #             "NAME": "weistZum",
        #             "TYPE": "Role",
        #             "client_cardinality": "0..1",
        #             "documentation": "Eine 'Lagebezeichnung mit Hausnummer' weist zum 'Turm'.",
        #             "supplier": "...:AX_Turm"
        #         },
        #         {
        #             "NAME": "zeigtAuf",
        #             "TYPE": "Role",
        #             "client_cardinality": "0..*",
        #             "documentation": "'Turm' zeigt auf eine 'Lagebezeichnung mit Hausnummer'.",
        #             "supplier": "...:AX_LagebezeichnungMitHausnummer"
        #         }
        #     ]
        # }

        role1 = rec['roles'][0]
        role2 = rec['roles'][1]

        type1 = role1['supplier'].split(':')[-1]
        type2 = role2['supplier'].split(':')[-1]

        cls1 = self.find_node(type1)
        cls2 = self.find_node(type2)

        if cls2 and not role1['NAME'].startswith('$') and cls2.attributes is not None:
            cls2.attributes.append(self.parse_object(role1, cls2))
        if cls1 and not role2['NAME'].startswith('$') and cls1.attributes is not None:
            cls1.attributes.append(self.parse_object(role2, cls1))


##

class Parser7(Parser):
    engine: sa.Engine

    def parse(self, paths):
        for path in paths:
            self.engine = sa.create_engine(f'sqlite:///' + path)
            self.build_from_sqlite()
        return self.finalize()

    def select(self, table):
        with self.engine.begin() as conn:
            sel = sa.text(f'SELECT * FROM {table}')
            return list(conn.execute(sel).mappings().all())

    def build_from_sqlite(self):

        nodes_by_uid = {}
        nodes_by_gid = {}

        for rec in self.select('t_object'):
            if rec['Alias']:
                continue

            node = Node(name=rec['Name'], doc=self.get_doc(rec))
            self.nodes.append(node)

            node.hname = self.get_hname(node)
            node.name = to_name(node.name)

            node.Package_ID = rec['Package_ID']

            nodes_by_uid[rec['Object_ID']] = node
            nodes_by_gid[rec['ea_guid']] = node

            if rec['Object_Type'] == 'Package':
                node.T = T_CATEGORY
                continue

            stereo = (rec['Stereotype'] or '').lower()

            if rec['Object_Type'] == 'Enumeration' or stereo in {'enumeration', 'codelist'}:
                node.T = T_ENUM
                node.values = {}
                continue

            if rec['Object_Type'] == 'Class' and stereo == 'union':
                node.T = T_UNION
                node.attributes = []
                continue

            if rec['Object_Type'] == 'Class':
                node.T = T_CLASS
                node.attributes = []
                node.supers = []
                node.pSuperNames = []
                continue

        for rec in self.select('t_objectproperties'):
            if rec['Property'] == 'AAA:Kennung' and rec['Value']:
                node = nodes_by_uid.get(rec['Object_ID'])
                if node:
                    node.uid = rec['Value']

        package_uid_to_gid = {}

        for rec in self.select('t_package'):
            package_uid_to_gid[rec['Package_ID']] = rec['ea_guid']

        for node in self.nodes:
            pkg_gid = package_uid_to_gid.get(popattr(node, 'Package_ID'))
            if pkg_gid:
                pkg_node = nodes_by_gid.get(pkg_gid)
                if pkg_node:
                    node.pParent = pkg_node

        for rec in self.select('t_attribute'):
            node = nodes_by_uid.get(rec['Object_ID'])
            if node:
                if node.T in {T_CLASS, T_UNION}:
                    a = Node(name=rec['Name'], doc=self.get_doc(rec), pParent=node)
                    self.set_type_from_record(a, rec)
                    self.set_cardinality_from_record(a, rec)
                    node.attributes.append(a)
                    self.nodes.append(a)
                if node.T == T_ENUM:
                    self.add_enum_value(node, rec['Default'], rec['Name'])

        for rec in self.select('t_connector'):
            so = nodes_by_uid.get(rec['Start_Object_ID'])
            eo = nodes_by_uid.get(rec['End_Object_ID'])

            if so and eo and so.T == eo.T == T_CLASS:
                if rec['Connector_Type'] == 'Generalization':
                    so.pSuperNames.append(eo.name)
                    continue

                if rec['Connector_Type'] == 'Association':
                    """
                        "SourceCard": "0..*",
                        "SourceRole": "zeigtAuf",
                        "SourceRoleNote": "'Turm' zeigt auf eine 'Lagebezeichnung mit Hausnummer'.",
                        "DestRole": "weistZum",
                        "DestRoleNote": "Eine 'Lagebezeichnung mit Hausnummer' weist zum 'Turm'.",
                        "Start_Object_ID": 3678,
                        "End_Object_ID": 3511,
                    """
                    if rec['SourceRole']:
                        a = Node(name=rec['SourceRole'], doc=rec['SourceRoleNote'], type=so.name, pParent=eo)
                        self.set_cardinality_from_string(a, rec['SourceCard'])
                        eo.attributes.append(a)
                        self.nodes.append(a)

                    if rec['DestRole']:
                        b = Node(name=rec['DestRole'], doc=rec['DestRoleNote'], type=eo.name, pParent=so)
                        so.attributes.append(b)
                        self.nodes.append(b)


class CatParser:
    """Parser for RR cat files."""

    def parse(self, path):
        with open(path, 'rb') as fp:
            text = fp.read().decode('latin-1')
        self.tokenize(text)
        return self.parse_sequence()

    re_token = r'''(?x)
        ( [()] )
        |
        ( [_a-zA-Z] \w* )
        |
        ( 
            " (?: \\. | [^"] )* " 
            | 
            [^()\s]+ 
        )
    '''

    tokens = []
    token_pos = 0

    def tokenize(self, text):
        docstring_buf = []
        self.tokens = []

        for n, ln in enumerate(text.split('\n'), 1):
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith('|'):
                docstring_buf.append(ln[1:])
                continue
            if docstring_buf:
                s = '\n'.join(p for p in docstring_buf if p).strip() or ' '
                self.tokens.append(('', '', s))
                docstring_buf = []
            for br, name, val in re.findall(self.re_token, ln):
                if val.startswith('"'):
                    # decode a string, don't allow empty strings
                    val = val[1:-1].replace('\\', '') or ' '
                self.tokens.append((br, name, val))

    def tok(self):
        return self.tokens[self.token_pos]

    def pop(self):
        self.token_pos += 1

    def eof(self):
        return self.token_pos >= len(self.tokens)

    ##

    def parse_sequence(self):
        items = []
        while not self.eof():
            br, name, val = self.tok()
            if br == ')':
                self.pop()
                break
            items.append(self.parse_item())
        return items

    def parse_item(self):
        br, name, val = self.tok()
        if val:
            self.pop()
            return val

        if name in {'TRUE', 'FALSE'}:
            self.pop()
            return name == 'TRUE'

        if br == '(':
            self.pop()
            br, name, val = self.tok()
            if name == 'list':
                # (list ...
                return self.parse_list()
            if name == 'object':
                # (object ...
                return self.parse_object()
            if name == 'value':
                # (value ...
                return self.parse_value()

            # (val val...)
            return self.parse_sequence()

        raise SyntaxError(f'invalid token {br=} {name=} {val=}')

    def parse_list(self):
        # e.g. (list Attribute_Set (object... (object...

        self.pop()  # list
        self.pop()  # type

        return self.parse_sequence()

    def parse_object(self):
        # e.g. (object ClassAttribute "Sonstiges" attr val attr val
        # e.g. (object Attribute

        rec = {}

        self.pop()  # object

        br, name, val = self.tok()
        rec['TYPE'] = name
        self.pop()

        br, name, val = self.tok()
        if val:
            rec['NAME'] = val
            self.pop()

        # evtl. more strings after name, ignore them
        while not self.eof():
            br, name, val = self.tok()
            if not val:
                break
            self.pop()

        while not self.eof():
            br, name, val = self.tok()
            if br == ')':
                self.pop()
                break
            self.pop()
            rec[name] = self.parse_item()

        return rec

    def parse_value(self):
        # e.g. (value Text "30000")

        self.pop()  # value
        self.pop()  # type

        val = self.parse_item()
        self.pop()  # )

        return val


class PythonGenerator:
    unknownTypes = set()
    knownTypes = set()
    nameToNode = {}
    keyToNode = {}
    seen = set()
    metadata = {}
    py = []

    def __init__(self, nodes, version: str):
        self.nodes = nodes
        self.version = version

    def build(self):
        self.knownTypes = set(
            node.name
            for node in self.nodes
            if node.T in {T_CLASS, T_ENUM, T_UNION}
        )

        self.nameToNode = {node.name: node for node in self.nodes}
        self.keyToNode = {node.key: node for node in self.nodes}

        nodes = sorted(self.nodes, key=lambda n: n.name)
        self.make_nodes(nodes)

        py = nl(self.py)

        py = re.sub(r'(\n\w+: TypeAlias)', '\n\n\\1', py)
        py = re.sub(r'(\nclass )', '\n\n\\1', py)

        py = nl([
            PY_HEAD,
            *[f'{k}: TypeAlias = {v}' for k, v in sorted(STD_TYPES.items())],
            '',
            '',
            *[f'class {k}: ...' for k in sorted(self.unknownTypes)],
            '',
            '',
            py,
            '',
            '',
            'METADATA = {',
            json_dict_body(self.metadata, TAB),
            '}',
            '',

        ])

        return py.replace('<VERSION>', self.version)

    def make_nodes(self, nodes):
        for ts in T_UNION, T_CATEGORY, T_ENUM, T_CLASS:
            for node in nodes:
                if node.T == ts:
                    self.make_node(node)

    def make_node(self, node):
        if node.name not in self.seen:
            self.seen.add(node.name)
            fn = getattr(self, 'make_' + node.T)
            fn(node)
            self.make_metadata(node)

    def make_union(self, node):
        items = sorted(set(self.get_type(a.type) for a in node.attributes))
        typ = items[0] if len(items) == 1 else 'Union[' + comma(items) + ']'

        self.py.append(f'{node.name}: TypeAlias = {typ}')
        self.py.append(self.get_docstring(node, '', False))

    def make_category(self, node):
        self.py.append(f'class {node.name}(Category):')
        self.py.append(self.get_docstring(node, TAB, True))

    def make_enum(self, node):
        self.py.append(f'class {node.name}(Enumeration):')
        self.py.append(self.get_docstring(node, TAB, True))
        self.py.append('')
        self.py.append(f'{TAB}VALUES = {{')
        self.py.append(json_dict_body(node.values, TAB2))
        self.py.append(f'{TAB}}}')

    def make_class(self, node):
        node.attributes = node.attributes or []

        super_types = []

        for super_node in (node.supers or []):
            self.make_node(super_node)
            super_types.append(self.get_type(super_node.name, quoted=False))

        cls = f'class {node.name}'
        if super_types:
            cls += '(' + comma(super_types) + ')'
        else:
            cls += '(Object)'

        self.py.append(cls + ':')
        self.py.append(self.get_docstring(node, TAB, True))

        if node.name == 'AA_REO':
            self.py.append('')
            self.py.append(f'{TAB}geom: str')

        for a in sorted(node.attributes, key=lambda a: a.name):
            typ = self.get_type(a.type)
            if a.list:
                typ = f'list[{typ}]'
            if a.optional:
                typ = f'Optional[{typ}]'
            self.py.append('')
            self.py.append(f"{TAB}{to_name(a.name)}: {typ}")
            self.py.append(self.get_docstring(a, TAB, False))

    def make_metadata(self, node):
        d = {
            'kind': node.T,
            'name': node.name,
            'uid': node.uid or '',
            'key': node.key or '',
            'title': node.hname or '',
        }

        if node.T == T_CLASS:
            d.update(self.make_class_metadata(node))

        self.metadata[node.name] = d

    def make_class_metadata(self, node):
        d = {}

        d['kind'] = 'object' if node.is_aa else 'struct'
        d['geom'] = 1 if node.is_reo else 0
        d['attributes'] = []

        d['supers'] = [sup.name for sup in node.supers]

        for a in node.attributes:
            d['attributes'].append({
                'name': a.name,
                'title': a.hname or '',
                'type': a.type,
                'list': 1 if a.list else 0,
            })

        return d

    def get_type(self, typ, quoted=True):
        if not typ:
            return 'Any'

        if hasattr(__builtins__, typ):
            return typ

        if typ in STD_TYPES or typ in self.knownTypes:
            return quote(typ) if quoted else typ

        self.unknownTypes.add(typ)
        return quote(typ) if quoted else typ

    def get_docstring(self, node, indent, prepend_name):
        name = node.hname or node.name or ' '

        if node.doc:
            s = node.doc
            if prepend_name and name:
                s = name + '\n\n' + s
        else:
            s = name

        if s.endswith('"'):
            s += ' '
        return wrap_indent(Q3 + s + Q3, indent)


##


def popattr(obj, attr, default=None):
    return obj.__dict__.pop(attr, default)


def wrap_indent(s, indent):
    return nl(
        nl(indent + ln for ln in textwrap.wrap(p.strip(), WRAP_WIDTH))
        for p in s.split('\n')
    )


def quote(s):
    return "'" + (s or '') + "'"


_UID_DE_TRANS = {
    ord('ä'): 'ae',
    ord('ö'): 'oe',
    ord('ü'): 'ue',
    ord('ß'): 'ss',
    ord('Ä'): 'Ae',
    ord('Ö'): 'Oe',
    ord('Ü'): 'Ue',
}


def to_name(s):
    if not s:
        return ''
    s = str(s)
    if re.match(r'^[A-Za-z_][A-Za-z_0-9]*$', s):
        return s
    s = s.strip().translate(_UID_DE_TRANS)
    s = re.sub(r'\W+', '_', s).strip('_')
    if not s:
        return '_'
    if s[0].isdigit():
        s = '_' + s
    return s


def json_dict_body(d, indent):
    js = json.dumps(d, indent=len(TAB), ensure_ascii=False).split('\n')[1:-1]
    ind = ' ' * (len(indent) - len(TAB))
    return nl(ind + p for p in js)


comma = ', '.join
nl = '\n'.join

##

if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
