import gws.types as t


# class ProxyType(t.Enum):
#     DefaultProxy = "DefaultProxy"
#     Socks5Proxy = "Socks5Proxy"
#     HttpProxy = "HttpProxy"
#     HttpCachingProxy = "HttpCachingProxy"
#     FtpCachingProxy = "FtpCachingProxy"
#
#
# class ProxyConfig(t.Config):
#     enabled: bool = True
#     host: str
#     password: str = ''
#     port: int
#     type: ProxyType = 'DefaultProxy'
#     user: str = ''
#
#
# class Config(t.Config):
#     """Bundled QGIS configuration"""
#
#     #: QGIS_DEBUG (environment)
#     debug: int = 0
#     #: QGIS_SERVER_LOG_LEVEL (environment)
#     serverLogLevel: int = 2
#     #: QGIS_SERVER_CACHE_SIZE (environment)
#     serverCacheSize: int = 10000000
#     #: MAX_CACHE_LAYERS (environment)
#     maxCacheLayers: int = 4000
#     #: searchPathsForSVG (ini setting)
#     searchPathsForSVG: t.Optional[t.List[t.dirpath]]
#     # #: proxy configuration (ini setting)
#     # proxy: t.Optional[ProxyConfig]
