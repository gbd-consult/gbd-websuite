from . import image, types


class Config(image.Config):
    display: types.DisplayMode = 'tile'  #: layer display mode


class ImageTile(image.Image):
    def configure(self):
        super().configure()

        # with meta=1 MP will request the same tile multiple times
        # meta=4 is more efficient, however, meta=1 yields the first tile faster
        # which is crucial when browsing non-cached low resoltions
        # so, let's use 1 as default, overridable in the config
        #
        # @TODO make MP cache network requests

        self.grid.metaSize = self.grid.metaSize or 1
