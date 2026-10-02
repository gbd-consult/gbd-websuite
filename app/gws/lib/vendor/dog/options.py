class Options:
    docRoots: list[str] = []
    """Documentation root directories."""

    outputDir: str = ''
    """Output directory."""

    docPatterns: list[str] = ['*.doc.md']
    """Shell patterns for documentation files."""

    assetPatterns: list[str] = ['*.svg', '*.png']
    """Shell patterns for asset files."""

    excludeRegex: str = ''
    """Paths matching this regex will be excluded."""

    debug: bool = False
    """Debug/verbose mode."""

    fileSplitLevel: dict = {}
    """Split levels for output files (dict keyed by sid)."""

    tocDepth: dict = {}
    """How deep the TOC shows a section (dict keyed by sid)."""

    pageTemplate: str = ''
    """Jump template for HTML pages."""

    pageTemplateArgs: dict = {}
    """Additional data for use in the page template."""

    webRoot: str = ''
    """Prefix for all URLs."""

    staticDir: str = '_static'
    """Web directory for static files."""

    extraAssets: list[str] = []
    """Extra assets to be copied to the static dir."""

    includeTemplate: str = ''
    """Jump template to include in every section."""

    serverHost: str = '0.0.0.0'
    """Live server hostname."""

    serverPort: int = 5500
    """Live server port."""

    title: str = ''
    """Documentation title."""

    subTitle: str = ''
    """Documentation subtitle."""

    pdfPageTemplate: str = ''
    """Jump template for PDF."""

    pdfOptions: dict = {}
    """Options for wkhtmltopdf."""

    blendChars: str = ''
    """Characters that split and join words in the search index."""

    extraChars: str = ''
    """Characters treated as part of a word in the search index."""
