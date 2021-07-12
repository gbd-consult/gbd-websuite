"""Qgis Project API, XML-based."""

import bs4

import gws
import gws.types as t


def from_path(path: str) -> 'Project':
    return from_string(gws.read_file(path), path)


def from_string(xml: str, path: str = None) -> 'Project':
    return Project(xml, path or '')


class Project:
    def __init__(self, xml: str, path: str):
        self.bs = bs4.BeautifulSoup(xml, 'lxml-xml')
        self.path = path

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError('no path')
        gws.write_file(path, str(self.bs))
