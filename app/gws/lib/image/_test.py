"""Tests for the image module."""
import os

import gws
import gws.test.util as u
import gws.lib.image as image

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


# creates the images to compare with using PIL
def test_create_images():
    red = PIL.Image.new('RGBA', (50, 50), (255, 0, 0, 255))
    red.save('/tmp/red.png')

    red_crop = red.crop((10, 10, 30, 30))
    red_crop.save('/tmp/red-crop.png')

    red_box = PIL.Image.new('RGBA', (50, 50), (255, 0, 0, 255))
    draw = PIL.ImageDraw.Draw(red_box)
    draw.rectangle((0, 0, 49, 49), outline=(0, 255, 0, 255))
    red_box.save('/tmp/red-box.png')

    x = PIL.Image.new('RGBA', (25, 25), (255, 255, 255, 255))
    draw = PIL.ImageDraw.Draw(x)
    draw.line((0, 12, 24, 12), fill=(0, 255, 0, 255))
    draw.line((12, 0, 12, 24), fill=(0, 255, 0, 255))
    x.save('/tmp/x.png')

    red_paste = PIL.Image.new('RGBA', (50, 50), (255, 0, 0, 255))
    red_paste.paste(x, (12, 12))
    red_paste.save('/tmp/red-paste.png')

    red_rotate = red.rotate(-15)
    red_rotate.save('/tmp/red-rotate.png')

    red_text = PIL.Image.new('RGBA', (50, 50), (255, 0, 0, 255))
    draw = PIL.ImageDraw.Draw(red_text)
    font = PIL.ImageFont.load_default()
    draw.multiline_text((10, 10), 'FOOBAR', font=font, fill=(0, 255, 0, 255))
    red_text.save('/tmp/red-text.png')

    red_compose_size_difference = PIL.Image.new('RGBA', (50, 50), (255, 0, 0, 255))
    x = x.resize((50, 50), resample=PIL.Image.BICUBIC)
    red_compose_size_difference.alpha_composite(x)
    red_compose_size_difference.save('/tmp/red-compose-size-diff.png')

    red_compose_float = PIL.Image.new('RGBA', (50, 50), (255, 0, 0, 255))
    x.putalpha(int(255 * 0.5))
    red_compose_float = PIL.Image.alpha_composite(red_compose_float, x)
    red_compose_float.save('/tmp/red-compose-float.png')

    blue = PIL.Image.new('RGBA', (50, 50), (0, 0, 255, 255))
    blue.save('/tmp/blue.png')
    red_compose = PIL.Image.alpha_composite(red, blue)
    red_compose.save('/tmp/red-compose.png')

    assert True


# https://i-converter.com/files/png-to-url used to get the url
red = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TpaItCnYQcchQnSyIFXHUKhShQqgVWnUwufQLmjQkKS6OgmvBwY/FqoOLs64OroIg+AHi7OCk6CIl/q8ptIjx4Lgf7+497t4BQr3MNKtrAtB020wl4mImuyoGXiFgAH2IISQzy5iTpCQ8x9c9fHy9i/Is73N/jpCasxjgE4lnmWHaxBvE05u2wXmfOMyKskp8Tjxu0gWJH7muuPzGudBkgWeGzXRqnjhMLBY6WOlgVjQ14iniiKrplC9kXFY5b3HWylXWuid/YTCnryxzneYIEljEEiSIUFBFCWXYiNKqk2IhRftxD/9w0y+RSyFXCYwcC6hAg9z0g//B726tfGzSTQrGge4Xx/kYBQK7QKPmON/HjtM4AfzPwJXe9lfqwMwn6bW2FjkC+reBi+u2puwBlzvA0JMhm3JT8tMU8nng/Yy+KQsM3gK9a25vrX2cPgBp6ip5AxwcAmMFyl73eHdPZ2//nmn19wNcpnKe/XiMrwAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAAd0SU1FB+gCEAwaAksjVSMAAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAAARElEQVRo3u3PMREAAAgEoNf+nbWFgwcNqEkmD3SeEBEREREREREREREREREREREREREREREREREREREREREREREREbmxB7ECYhgv8rEAAAAASUVORK5CYII='


def test_from_size():
    img = image.from_size((50, 50), color=(255, 0, 0, 255))
    img2 = image.from_path('/tmp/red.png')
    assert img.compare_to(img2) == 0


def test_from_bytes():
    file = open('/tmp/red.png', 'rb')
    byt = file.read()
    img = image.from_bytes(byt)
    img2 = image.from_path('/tmp/red.png')
    assert img.compare_to(img2) == 0


def test_from_raw_data():
    img = image.from_data_url(red)
    arr = img.to_array()
    arr = arr.tobytes()
    img2 = image.from_raw_data(arr, 'RGBA', (50, 50))
    assert img.compare_to(img2) == 0


def test_from_path():
    img = image.from_path('/tmp/red.png')
    img2 = image.from_data_url(red)
    assert img.compare_to(img2) == 0


# https://i-converter.com/files/png-to-url used to get the url
def test_from_data_url():
    img = image.from_data_url(red)
    img2 = image.from_path('/tmp/red.png')
    assert img.compare_to(img2) == 0


def test_from_data_url_exception():
    with u.raises(Exception, match=f'invalid data url'):
        img = image.from_data_url('foobar')


# not implemented
def test_from_svg():
    with u.raises(Exception):
        img = image.from_svg('test', (50, 50))


def test_size():
    img = image.from_path('/tmp/red.png')
    assert img.size() == (50, 50)


def test_resize():
    img = image.from_size((100, 100))
    img.resize((50, 50))
    assert img.size() == (50, 50)


def test_rotate():
    img = image.from_data_url(red)
    img = img.rotate(-15)
    img2 = image.from_path('/tmp/red-rotate.png')
    # sometimes not exactly 100% similar due to antialiasing when rotating
    assert img.compare_to(img2) == 0


def test_crop():
    img = image.from_data_url(red)
    img = img.crop((10, 10, 30, 30))
    img2 = image.from_path('/tmp/red-crop.png')
    assert img.compare_to(img2) == 0


def test_paste():
    img = image.from_data_url(red)
    x = image.from_path('/tmp/x.png')
    img = img.paste(x, where=(12, 12))
    img2 = image.from_path('/tmp/red-paste.png')
    assert img.compare_to(img2) == 0


# not right
def test_compose_float():
    img = image.from_data_url(red)
    img2 = image.from_path('/tmp/x.png')
    img = img.compose(img2, opacity=0.5)
    img3 = image.from_path('/tmp/red-compose-float.png')
    assert img.compare_to(img3) == 0


def test_compose_size_difference():
    img = image.from_data_url(red)
    img2 = image.from_path('/tmp/x.png')
    img = img.compose(img2)
    img3 = image.from_path('/tmp/red-compose-size-diff.png')
    assert img.compare_to(img3) == 0


def test_compose():
    img = image.from_data_url(red)
    img2 = image.from_path('/tmp/blue.png')
    img3 = image.from_path('/tmp/red-compose.png')
    img = img.compose(img2)
    assert img.compare_to(img3) == 0


def test_to_bytes():
    img = image.from_size((2.0, 2.0))
    img.to_path('/tmp/img.png')
    file = open('/tmp/img.png', 'rb')
    byt = file.read()
    assert img.to_bytes('image/png') == byt


def test_to_path():
    img = image.from_size((10.0, 10.0))
    assert '/tmp/img.png' == img.to_path('/tmp/img.png')
    assert os.path.isfile('/tmp/img.png')


def test_to_array():
    img = image.from_size((2.0, 2.0), color=(1, 2, 3, 4))
    assert img.to_array().tolist() == ([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])


def test_add_text():
    img = image.from_data_url(red)
    img.add_text('FOOBAR', 10, 10, (0, 255, 0, 255))
    img2 = image.from_path('/tmp/red-text.png')
    assert img.compare_to(img2) == 0


# box is not drawn around all edges
def test_add_box():
    img = image.from_data_url(red)
    img = img.add_box((0, 255, 0, 255))
    img2 = image.from_path('/tmp/red-box.png')
    assert img.compare_to(img2) == 0
