"""MBTiles provider."""


import gws


class Config(gws.Config):
    """Access to an MBTiles file."""
    
    path: gws.FilePath
    """Path to the MBTiles file."""



class Object(gws.Node):
    """MBTiles file provider."""

    path: str
    """Path to the MBTiles file."""

    def configure(self):
        self.path = self.cfg('path')

    def cache_hash(self):
        """Return a hash that identifies the data of the provider.

        Returns:
            A hash of the file path.
        """
        return gws.u.sha256([self.path])
