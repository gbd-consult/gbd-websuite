"""MBTiles provider."""


import gws


class Config(gws.Config):
    """MBTiles provider configuration."""
    
    path: gws.FilePath
    """List of image file paths."""



class Object(gws.ServiceProvider):
    path: str

    def configure(self):
        self.path = self.cfg('path')

    def cache_hash(self):
        return gws.u.sha256([self.path])
