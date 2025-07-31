"""MBTiles provider."""


import gws


class Config(gws.Config):
    """MBTiles provider configuration."""
    
    path: gws.FilePath
    """List of image file paths."""



class Object(gws.Node):
    path: str

    def configure(self):
        self.path = self.cfg('path')
