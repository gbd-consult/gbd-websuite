"""MBTiles provider."""


import gws


class Config(gws.Config):
    """Access to an MBTiles file."""
    
    path: gws.FilePath
    """Path to the MBTiles file."""



class Object(gws.Node):
    path: str

    def configure(self):
        self.path = self.cfg('path')

    def cache_hash(self):
        return gws.u.sha256([self.path])
