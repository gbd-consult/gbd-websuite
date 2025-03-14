"""Tests for the core module."""

import gws
import gws.lib.metadata.core as metadata_core
import gws.test.util as u


def test_from_dict():
    """Test creating metadata from a dictionary."""
    data = {
        'title': 'Test Title',
        'abstract': 'Test Abstract',
        'keywords': ['test', 'metadata'],
        'language': 'en'
    }

    md = metadata_core.from_dict(data)

    assert isinstance(md, gws.Metadata)
    assert md.title == 'Test Title'
    assert md.abstract == 'Test Abstract'
    assert md.keywords == ['test', 'metadata']
    assert md.language == 'en'


def test_from_args():
    """Test creating metadata from keyword arguments."""
    md = metadata_core.from_args(
        title='Test Title',
        abstract='Test Abstract',
        keywords=['test', 'metadata'],
        language='en'
    )

    assert isinstance(md, gws.Metadata)
    assert md.title == 'Test Title'
    assert md.abstract == 'Test Abstract'
    assert md.keywords == ['test', 'metadata']
    assert md.language == 'en'


def test_from_config():
    """Test creating metadata from a configuration object."""
    config = gws.Config({
        'title': 'Test Title',
        'abstract': 'Test Abstract',
        'keywords': ['test', 'metadata'],
        'language': 'en'
    })

    md = metadata_core.from_config(config)

    assert isinstance(md, gws.Metadata)
    assert md.title == 'Test Title'
    assert md.abstract == 'Test Abstract'
    assert md.keywords == ['test', 'metadata']
    assert md.language == 'en'


def test_from_props():
    """Test creating metadata from properties."""
    props = gws.Props({
        'title': 'Test Title',
        'abstract': 'Test Abstract',
        'keywords': ['test', 'metadata'],
        'language': 'en'
    })

    md = metadata_core.from_props(props)

    assert isinstance(md, gws.Metadata)
    assert md.title == 'Test Title'
    assert md.abstract == 'Test Abstract'
    assert md.keywords == ['test', 'metadata']
    assert md.language == 'en'


def test_props():
    """Test extracting properties from a Metadata object."""
    props = gws.Props({
        'title': 'Test Title',
        'abstract': 'Test Abstract',
        'keywords': ['test', 'metadata'],
        'language': 'en'
    })

    md = metadata_core.from_props(props)

    props = metadata_core.props(md)

    assert isinstance(props, gws.Props)
    assert props.get('title') == 'Test Title'
    assert props.get('abstract') == 'Test Abstract'
    assert props.get('keywords') == ['test', 'metadata']
    assert props.get('language') == 'en'


def test_set_value():
    """Test setting a value in a Metadata object."""
    md = gws.Metadata()

    md = metadata_core.set_value(md, 'title', 'New Title')
    assert md.title == 'New Title'

    md = metadata_core.set_value(md, 'keywords', ['new', 'keywords'])
    assert md.keywords == ['new', 'keywords']


def test_set_default():
    """Test setting a default value in a Metadata object."""
    md = gws.Metadata({'title': 'Existing Title'})

    # Should not change existing value
    result = metadata_core.set_default(md, 'title', 'Default Title')
    assert result.title == 'Existing Title'

    # Should set new value
    result = metadata_core.set_default(md, 'foo', 'bar')
    assert result.foo == 'bar'


def test_check_list_normalization():
    """Test that check normalizes list values."""
    md = gws.Metadata({'keywords': 'single_keyword'})

    md = metadata_core.check(md)
    assert isinstance(md.keywords, list)
    assert md.keywords == ['single_keyword']

    md = gws.Metadata({'keywords': ['keyword1', 'keyword2']})

    md = metadata_core.check(md)
    assert isinstance(md.keywords, list)
    assert md.keywords == ['keyword1', 'keyword2']


def test_check_language_processing():
    """Test that check processes language codes correctly."""
    md = gws.Metadata({'language': 'en'})

    md = metadata_core.check(md)
    assert md.language == 'en'
    assert md.language3 == 'eng'
    assert md.languageBib == 'eng'


def test_check_inspire_theme_processing():
    """Test that check processes INSPIRE themes correctly."""
    md = gws.Metadata({
        'inspireTheme': 'ad',
        'language': 'en'
    })

    md = metadata_core.check(md)
    assert md.inspireTheme == 'ad'
    assert md.inspireThemeName == 'Addresses'
    assert md.inspireThemeNameEn == 'Addresses'

    # Test with German language
    md = gws.Metadata({
        'inspireTheme': 'ad',
        'language': 'de'
    })

    md = metadata_core.check(md)
    assert md.inspireThemeName == 'Adressen'
    assert md.inspireThemeNameEn == 'Addresses'


def test_merge_basic():
    """Test merging basic metadata objects."""
    md1 = gws.Metadata({
        'title': 'Title 1',
        'abstract': 'Abstract 1'
    })

    md2 = gws.Metadata({
        'title': 'Title 2',
        'keywords': ['keyword2']
    })

    merged = metadata_core.merge(md1, md2)

    assert merged.title == 'Title 2'  # Second value overwrites
    assert merged.abstract == 'Abstract 1'  # Kept from first
    assert merged.keywords == [
        'keyword2']  # Added from second


def test_merge_with_empty_values():
    """Test merging with empty values."""
    md1 = gws.Metadata({
        'title': 'Title 1',
        'abstract': ''
    })

    md2 = gws.Metadata({
        'title': '',
        'abstract': 'Abstract 2'
    })

    merged = metadata_core.merge(md1, md2)

    assert merged.title == 'Title 1'  # Empty value doesn't overwrite
    assert merged.abstract == 'Abstract 2'  # Empty value gets replaced


def test_merge_with_none_objects():
    """Test merging with None objects."""
    md1 = gws.Metadata({
        'title': 'Title 1'
    })

    merged = metadata_core.merge(md1, None)
    assert merged.title == 'Title 1'

    merged = metadata_core.merge(None, md1)
    assert merged.title == 'Title 1'

    merged = metadata_core.merge(None, None)
    assert isinstance(merged, gws.Metadata)


def test_merge_with_extend_lists():
    """Test merging with extend_lists option."""
    md1 = gws.Metadata({
        'keywords': ['key1', 'key2']
    })

    md2 = gws.Metadata({
        'keywords': ['key3', 'key4']
    })

    # Without extend_lists
    merged = metadata_core.merge(md1, md2)
    assert merged.keywords == ['key3',
                               'key4']  # Second overwrites

    # With extend_lists
    merged = metadata_core.merge(md1, md2, extend_lists=True)
    assert merged.keywords == ['key1', 'key2', 'key3',
                               'key4']  # Lists combined


def test_merge_multiple_objects():
    """Test merging multiple metadata objects."""
    md1 = gws.Metadata({'title': 'Title 1'})
    md2 = gws.Metadata({'abstract': 'Abstract 2'})
    md3 = gws.Metadata({'keywords': ['key3']})

    merged = metadata_core.merge(md1, md2, md3)

    assert merged.title == 'Title 1'
    assert merged.abstract == 'Abstract 2'
    assert merged.keywords == ['key3']
