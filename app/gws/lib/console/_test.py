import gws.lib.console

def test_text_table():
    data = [
        {
            "a": 1,
            "b": 2,
            "c": 3
        },
        {
            "a": 4,
            "b": 5,
            "c": 6
        },
        {
            "a": 7,
            "b": 8,
            "c": 9
        }
    ]
    assert gws.lib.console.text_table(data, None) == "1 | 2 | 3\n4 | 5 | 6\n7 | 8 | 9"
    assert gws.lib.console.text_table(data, 'auto') == "a | b | c\n---------\n1 | 2 | 3\n4 | 5 | 6\n7 | 8 | 9"
    assert gws.lib.console.text_table(data, 'auto', '#') == "a#b#c\n-----\n1#2#3\n4#5#6\n7#8#9"