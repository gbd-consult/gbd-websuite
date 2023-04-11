# import gws
# import gws.base.layer
# import gws.gis.ows
# import gws.types as t
#
# from . import provider as provider_module
# from . import search
#
# gws.ext.new.layer('wfs')
#
#
# class Config(gws.base.layer.vector.Config, provider_module.Config):
#     """WFS layer"""
#     pass
#
#
# class Object(gws.base.layer.vector.Object, gws.IOwsClient):
#     provider: provider_module.Object
#
#     def configure_source(self):
#         gws.gis.ows.client.configure_layers(self, provider_module.Object)
#         return True
#
#     def configure_metadata(self):
#         if not super().configure_metadata():
#             self.set_metadata(self.cfg('metadata'), self.provider.metadata)
#             return True
#
#     def configure_search(self):
#         if not super().configure_search():
#             return gws.gis.ows.client.configure_search(self, search.Object)
#
#     def get_features(self, bounds, limit=0):
#         features = self.provider.find_source_features(gws.SearchArgs(bounds=bounds, limit=limit), self.source_layers)
#         return [f.connect_to(self) for f in features]
