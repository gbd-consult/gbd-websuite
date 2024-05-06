"""Tests for the jsonx module."""

import gws
import gws.test.util as u
import gws.lib.jsonx as jsonx

_json_pretty = """{
    "additionalInfo": null,
    "address": {
        "city": "New York",
        "postalCode": 10021,
        "state": "NY",
        "streetAddress": "21 2nd Street"
    },
    "ficoScore": " > 640",
    "height": 62.4,
    "name": "John Smith",
    "phoneNumbers": [
        "212 555-1111",
        "212 555-2222"
    ],
    "remote": false
}"""

_json = """{
  "name":"John Smith",
  "address": {
    "streetAddress": "21 2nd Street",
    "city": "New York",
    "state": "NY",
    "postalCode": 10021
  },
  "phoneNumbers": [
    "212 555-1111",
    "212 555-2222"
  ],
  "additionalInfo": null,
  "remote": false,
  "height": 62.4,
  "ficoScore": " > 640"
}"""

_jsondict = {
    "name": "John Smith",
    "address": {
        "streetAddress": "21 2nd Street",
        "city": "New York",
        "state": "NY",
        "postalCode": 10021
    },
    "phoneNumbers": [
        "212 555-1111",
        "212 555-2222"
    ],
    "additionalInfo": None,
    "remote": False,
    "height": 62.4,
    "ficoScore": " > 640"
}


def test_form_path(tmp_path):
    path_json = tmp_path / "test.json"

    with open(str(path_json), "w") as f:
        f.write(_json)

    j = jsonx.from_path(str(path_json))
    assert j == _jsondict


def test_from_string():
    j = jsonx.from_string(_json)
    assert j == _jsondict


def test_to_path(tmp_path):
    path_json = tmp_path / "test.json"

    jsonx.to_path(str(path_json), _jsondict)

    with open(str(path_json), 'r') as f:
        assert f.read().replace(" ", "").replace("\n", "") == _json.replace(" ", "").replace("\n", "")


def test_to_string():
    s = jsonx.to_string(_jsondict)
    assert s.replace(" ", "").replace("\n", "") == _json.replace(" ", "").replace("\n", "")


def test_to_string_ascii():
    jsondict = {"FÖÖ": "BÄR"}
    s = jsonx.to_string(jsondict, ensure_ascii=False)
    assert s == '{"FÖÖ": "BÄR"}'


def test_to_string_ascii_escaped():
    jsondict = {"FÖÖ": "BÄR"}
    s = jsonx.to_string(jsondict)
    assert s == '{"F\\u00d6\\u00d6": "B\\u00c4R"}'


def test_to_string_default():
    def default(x):
        return {"foo": "bar"}

    class CustomObject:
        def __init__(self, name):
            self.name = name

    custom = CustomObject("Custom")
    jsondict = {"foo": custom}
    json_string = jsonx.to_string(jsondict, default=default)
    assert json_string == '{"foo": {"foo": "bar"}}'


def test_to_pretty_string():
    s = jsonx.to_pretty_string(_jsondict)
    assert s == _json_pretty
