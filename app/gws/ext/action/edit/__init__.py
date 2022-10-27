"""Backend for vector edit operations."""

import gws
import gws.common.action
import gws.gis.feature
import gws.tools.json2
import gws.web.error
import gws.tools.date
import gws.common.model
import gws.common.printer.types
import gws.common.printer.job
import gws.common.template

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Feature edit action"""
    printTemplate: t.Optional[t.ext.template.Config]  #: template for printed features


class GetFeaturesParams(t.Params):
    layerUid: str


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


class EditParams(t.Params):
    layerUid: str
    features: t.List[t.FeatureProps]


class EditResponse(t.Response):
    features: t.List[t.FeatureProps]
    failures: t.Optional[t.List[gws.common.model.AttributeValidationFailure]]


class PrintParams(t.Params):
    layerUid: str
    featureUids: t.List[str]
    printParams: gws.common.printer.types.PrintParamsWithTemplate


class Props(t.Props):
    type: t.Literal = 'edit'
    printTemplate: gws.common.template.TemplateProps


class Object(gws.common.action.Object):
    def configure(self):
        super().configure()
        self.print_template = None
        p = self.var('printTemplate')
        if p:
            self.print_template = self.create_child('gws.ext.template', p)

    def props_for(self, user):
        if not user.can_use(self):
            return None

        return {
            'type': self.type,
            'printTemplate': self.print_template.props if self.print_template else None,
        }

    def api_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get editable features"""

        layer: t.ILayer = req.require_layer(p.layerUid)
        if not layer.edit_access(req.user):
            return GetFeaturesResponse(features=[])

        features = layer.get_editable_features()

        for f in features:
            f.transform_to(layer.map.crs)
            f.apply_templates()
            f.apply_data_model()

        return GetFeaturesResponse(features=[f.props for f in features])

    def http_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> t.HttpResponse:
        res = self.api_get_features(req, p)
        return t.HttpResponse(mime='application/json', content=gws.tools.json2.to_string(res))

    def api_add_features(self, req: t.IRequest, p: EditParams) -> EditResponse:
        """Add features to the layer"""

        return self._handle('insert', req, p)

    def api_delete_features(self, req: t.IRequest, p: EditParams) -> EditResponse:
        """Delete features from the layer"""

        return self._handle('delete', req, p)

    def api_update_features(self, req: t.IRequest, p: EditParams) -> EditResponse:
        """Update features on the layer"""

        return self._handle('update', req, p)

    def _handle(self, op, req, p: EditParams):
        layer: t.ILayer = req.require_layer(p.layerUid)
        if not layer.edit_access(req.user):
            raise gws.web.error.Forbidden()

        src_features = []
        failures = []

        for f in p.features:
            f.attributes = f.attributes or []
            if layer.edit_data_model:
                if op == 'update':
                    fs = layer.edit_data_model.validate(f.attributes)
                    if fs:
                        failures.extend(fs)
                        continue

                f.attributes.append(t.Attribute(name='gws:user_login', value=req.user.attribute('login')))
                f.attributes.append(t.Attribute(name='gws:current_datetime', value=gws.tools.date.now()))
            src_features.append(f)

        dst_features = []

        if src_features:
            dst_features = layer.edit_operation(op, src_features)
            for f in dst_features:
                f.transform_to(layer.map.crs)
                f.apply_templates()
                f.apply_data_model()

        return EditResponse(features=[f.props for f in dst_features], failures=failures)

    def api_print(self, req: t.IRequest, p: PrintParams) -> gws.common.printer.job.StatusResponse:
        """Print features"""

        if not self.print_template:
            raise gws.web.error.NotFound()

        layer: t.ILayer = req.require_layer(p.layerUid)

        features = layer.get_editable_features()
        uids = set(p.featureUids)
        print_features = []

        for f in features:
            if f.full_uid in uids:
                f.transform_to(layer.map.crs)
                f.apply_templates()
                f.apply_data_model()
                print_features.append(f)

        if not print_features:
            gws.log.debug(f'no print features uids={uids!r}')
            raise gws.web.error.NotFound()

        pp = p.printParams
        pp.projectUid = p.projectUid
        pp.locale = p.locale
        pp.templateUid = self.print_template.uid
        pp.sections = []

        for feature in print_features:
            center = feature.shape.centroid
            pp.sections.append(gws.common.printer.types.PrintSection(
                center=[center.x, center.y],
                context=feature.template_context,
                items=[
                    gws.common.printer.types.PrintItemFeatures(
                        type='features',
                        features=[feature.props],
                        style=layer.style,
                    )
                ]
            ))

        job = gws.common.printer.job.start(req, pp)
        return gws.common.printer.job.status(job)
