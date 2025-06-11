import gws
import gws.base.metadata.core as mdc
import gws.test.util as u


def test_from_dict():
    data = {'title': 'Test Title', 'abstract': 'Test Abstract', 'keywords': ['bbb', 'aaa'], 'language': 'en'}

    md = mdc.from_dict(data)
    print('@@@@@@@@@@', md)

    assert isinstance(md, gws.Metadata)
    assert md.title == 'Test Title'
    assert md.abstract == 'Test Abstract'
    assert md.keywords == ['aaa', 'bbb']
    assert md.language == 'en'


def test_from_config():
    config = gws.Config({'title': 'Test Title', 'abstract': 'Test Abstract', 'keywords': ['bbb', 'aaa'], 'language': 'en'})

    md = mdc.from_config(config)

    assert isinstance(md, gws.Metadata)
    assert md.title == 'Test Title'
    assert md.abstract == 'Test Abstract'
    assert md.keywords == ['aaa', 'bbb']
    assert md.language == 'en'


def test_from_props():
    props = gws.Props({'title': 'Test Title', 'abstract': 'Test Abstract', 'keywords': ['bbb', 'aaa'], 'language': 'en'})

    md = mdc.from_props(props)

    assert isinstance(md, gws.Metadata)
    assert md.title == 'Test Title'
    assert md.abstract == 'Test Abstract'
    assert md.keywords == ['aaa', 'bbb']
    assert md.language == 'en'


def test_props():
    """Test extracting properties from a Metadata object."""
    props = gws.Props({'title': 'Test Title', 'abstract': 'Test Abstract', 'keywords': ['bbb', 'aaa'], 'language': 'en'})

    md = mdc.from_props(props)

    props = mdc.props(md)

    assert isinstance(props, gws.Props)
    assert props.get('title') == 'Test Title'
    assert props.get('abstract') == 'Test Abstract'
    assert props.get('keywords') == ['aaa', 'bbb']
    assert props.get('language') == 'en'


def test_check_language_processing():
    """Test that check processes language codes correctly."""
    md = mdc.from_dict({'language': 'de'})
    assert md.language == 'de'
    assert md.language3 == 'deu'
    assert md.languageBib == 'ger'


def test_check_inspire_theme_processing():
    """Test that check processes INSPIRE themes correctly."""
    md = mdc.from_dict({'inspireTheme': 'ac', 'language': 'en'})

    assert md.inspireTheme == 'ac'
    assert md.inspireThemeNameLocal == 'Atmospheric conditions'
    assert md.inspireThemeNameEn == 'Atmospheric conditions'

    md = mdc.from_dict({'inspireTheme': 'ac', 'language': 'de'})

    assert md.inspireThemeNameLocal == 'Atmosphärische Bedingungen'
    assert md.inspireThemeNameEn == 'Atmospheric conditions'


def test_merge_basic():
    """Test merging basic metadata objects."""
    md1 = mdc.from_dict(
        {
            'title': 'Title 1',
            'abstract': 'Abstract 1',
            'keywords': ['bbb', 'aaa'],
        }
    )

    md2 = mdc.from_dict(
        {
            'title': 'Title 2',
            'keywords': ['aaa', 'ccc'],
        }
    )

    merged = mdc.from_args(md1, md2)

    assert merged.title == 'Title 2'
    assert merged.abstract == 'Abstract 1'
    assert merged.keywords == ['aaa', 'bbb', 'ccc']


def test_merge_multiple_objects():
    md1 = mdc.from_dict({'title': 'Title 1'})
    md2 = mdc.from_dict({'abstract': 'Abstract 2'})
    md3 = mdc.from_dict({'keywords': ['key3']})

    merged = mdc.from_args(md1, md2, md3)

    assert merged.title == 'Title 1'
    assert merged.abstract == 'Abstract 2'
    assert merged.keywords == ['key3']
