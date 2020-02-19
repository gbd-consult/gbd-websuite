import bs4
import os
import sys
import re

DOC_ROOT = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.abspath(DOC_ROOT + '../../../app')

VERSION = open(DOC_ROOT + '/../../VERSION').read().strip()

_BASE_URL = 'https://gws.gbd-consult.de/doc/{release}/books/client-user/{lang}/overview'


def _abslink(href, base):
    if href.startswith(('http', '#')):
        return href
    if href.startswith('/'):
        return base + href
    return base + '/' + href


def make_help(lang):
    release = '.'.join(VERSION.split('.')[:-1])
    base = _BASE_URL.format(lang=lang, release=release)

    with open(DOC_ROOT + f'/../_build/books/client-user/{lang}/overview/help.html') as fp:
        html = fp.read()

    bs = bs4.BeautifulSoup(html, 'html.parser')

    for n in bs.find_all('link'):
        if n.get('href'):
            n['href'] = _abslink(n['href'], base)
    for n in bs.find_all('a'):
        if n.get('href'):
            n['href'] = _abslink(n['href'], base)
            n['target'] = '_blank'
    for n in bs.find_all('img'):
        if n.get('src'):
            n['src'] = _abslink(n['src'], base)

    for n in bs.find_all(['script', 'nav', 'footer']):
        n.extract()
    for n in bs.find_all(role='navigation'):
        n.extract()

    html = bs.prettify()
    html += """
        <style>
            .wy-nav-content-wrap {
                margin-left: 0 !important;
                background: none !important;
            }
            .wy-body-for-nav {
                background: none !important;
            }
        </style>
    """

    with open(DOC_ROOT + f'/../_build/help_{lang}.html', 'w') as fp:
        fp.write(html)


make_help('en')
make_help('de')
