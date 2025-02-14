import pytest
import gws
import gws.test.util as u
import gws.lib.htmlx as htmlx
from pathlib import Path


def test_render_to_pdf(tmp_path):
    html_content = "<html><body><h1>Test PDF</h1></body></html>"
    out_path = str(tmp_path / "output.pdf")
    result = htmlx.render_to_pdf(html_content, out_path)

    assert Path(out_path).exists(), "PDF file was not created"
    assert result == out_path


def test_render_to_pdf_with_page_size(tmp_path):
    html_content = "<html><body><h1>Test PDF</h1></body></html>"
    out_path = str(tmp_path / "output.pdf")
    page_size = (100, 200, gws.Uom.mm)
    result = htmlx.render_to_pdf(html_content, out_path, page_size=page_size)

    assert Path(out_path).exists(), "PDF file was not created with custom size"
    assert result == out_path


def test_render_to_pdf_with_margin(tmp_path):
    html_content = "<html><body><h1>Test PDF</h1></body></html>"
    out_path = str(tmp_path / "output.pdf")
    page_margin = (10, 10, 10, 10, gws.Uom.mm)
    result = htmlx.render_to_pdf(html_content, out_path, page_margin=page_margin)

    assert Path(out_path).exists(), "PDF file was not created with margins"
    assert result == out_path


def test_render_to_png(tmp_path):
    html_content = "<html><body><h1>Test PNG</h1></body></html>"
    out_path = str(tmp_path / "output.png")
    result = htmlx.render_to_png(html_content, out_path)

    assert Path(out_path).exists(), "PNG file was not created"
    assert result == out_path


def test_render_to_png_with_page_size(tmp_path):
    html_content = "<html><body><h1>Test PNG</h1></body></html>"
    out_path = str(tmp_path / "output.png")
    page_size = (500, 500, gws.Uom.px)
    result = htmlx.render_to_png(html_content, out_path, page_size=page_size)

    assert Path(out_path).exists(), "PNG file was not created with custom size"
    assert result == out_path


def test_render_to_png_with_margin(tmp_path):
    html_content = "<html><body><h1>Test PNG</h1></body></html>"
    out_path = str(tmp_path / "output.png")
    page_margin = [10, 10, 10, 10]
    result = htmlx.render_to_png(html_content, out_path, page_margin=page_margin)

    assert Path(out_path).exists(), "PNG file was not created with margins"
    assert result == out_path
