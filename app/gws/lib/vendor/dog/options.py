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
    """Split levels for output files."""

    pageTemplate: str = ''
    """Jump template for HTML pages."""

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
