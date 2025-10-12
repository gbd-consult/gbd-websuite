"""Tests for the pdf module."""

import gws
import gws.test.util as u
import gws.lib.pdf as p
import gws.lib.mime
import gws.lib.htmlx as htmlx


def make_test_pdfs(dir: str):
    s = '<div style="position: absolute; top: 0mm; left: 0mm;">A</div>'
    htmlx.render_to_pdf(s, f'{dir}/_a.pdf', page_size=(30, 30, gws.Uom.mm))

    s = '<div style="position: absolute; top: 10mm; left: 10mm;">B</div>'
    htmlx.render_to_pdf(s, f'{dir}/_b.pdf', page_size=(30, 30, gws.Uom.mm))

    s = '<div style="position: absolute; top: 20mm; left: 20mm;">C</div>'
    htmlx.render_to_pdf(s, f'{dir}/_c.pdf', page_size=(30, 30, gws.Uom.mm))

    p.concat([f'{dir}/_a.pdf', f'{dir}/_b.pdf', f'{dir}/_c.pdf'], f'{dir}/_abc.pdf')
    p.concat([f'{dir}/_b.pdf', f'{dir}/_c.pdf', f'{dir}/_a.pdf'], f'{dir}/_bca.pdf')

    p.overlay(f'{dir}/_abc.pdf', f'{dir}/_bca.pdf', f'{dir}/_abc_bca.pdf')


def test_all(tmp_path):
    make_test_pdfs(str(tmp_path))
    # @TODO: verify content
