import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './toolbar';
import * as sidebar from './sidebar';

let {Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Measure';

let EDITOR_HEIGHT = 90;

function dimensions(geom: ol.geom.Geometry, projection: ol.proj.Projection) {
    // @TODO: also provide point coordinates

    let type = geom.getType();

    if (type === 'LineString') {
        return {
            len: ol.Sphere.getLength(geom, {projection})
        }
    }

    if (type === 'Polygon') {
        let p = geom as ol.geom.Polygon;
        return {
            len: ol.Sphere.getLength(p.getLinearRing(0), {projection}),
            area: ol.Sphere.getArea(p, {projection}),
        }
    }

    if (type === 'Circle') {
        let c = geom as ol.geom.Circle,
            p = ol.geom.Polygon.fromCircle(c, 64, 0);

        return {
            ...dimensions(p, projection),
            radius: c.getRadius(),
        }
    }

    return {};
}

function formatLength(n) {
    if (!n || n < 0.01)
        return '';
    if (n >= 1e3)
        return (n / 1e3).toFixed(2) + ' km';
    if (n > 1)
        return n.toFixed(0) + ' m';
    return n.toFixed(2) + ' m';
}

function formatArea(n) {
    let sq = '\u00b2';

    if (!n || n < 0.01)
        return '';
    if (n >= 1e5)
        return (n / 1e6).toFixed(2) + ' km' + sq;
    if (n > 1)
        return n.toFixed(0) + ' m' + sq;
    return n.toFixed(2) + ' m' + sq;
}

function formattedDimensions(geom: ol.geom.Geometry, projection: ol.proj.Projection) {
    let d = dimensions(geom, projection);

    return {
        len: formatLength(d.len),
        area: formatArea(d.area),
        radius: formatLength(d.radius),
    }
}

class MeasureLayer extends gws.map.layer.FeatureLayer {
}

class MeasureFeature extends gws.map.Feature {
    labelTemplate: string;
    titleTemplate: string;
    title: string;

    constructor(map, args, titleTemplate, labelTemplate) {
        super(map, args);
        this.oFeature.getGeometry().on('change', () => this.update());
        this.titleTemplate = titleTemplate;
        this.labelTemplate = labelTemplate;
        this.update();
    }

    update() {
        let dims = formattedDimensions(
            this.oFeature.getGeometry(),
            this.map.projection);

        this.title = this.format(this.titleTemplate, dims);
        this.setLabel(this.format(this.labelTemplate, dims));
    }

    setLabelTemplate(s) {
        this.labelTemplate = s;
        this.update();
    }

    setTitleTemplate(s) {
        this.titleTemplate = s;
        this.update();
    }

    protected format(text, dims) {
        return (text || '').replace(/{(\w+)}/g, ($0, $1) => dims[$1] || '');

    }

}

abstract class MeasureTool extends gws.Controller implements gws.types.ITool {
    geometryType: string;
    geometryName: string;
    cssSelector: string;
    defaultLabel: string;

    drawFeature: MeasureFeature;

    start() {
        let master = this.app.controller(MASTER) as MeasureController,
            layer = master.getOrCreateLayer(),
            style = this.map.getStyleFromSelector(this.cssSelector);

        console.log('MEASURE_START');

        let draw = this.map.drawInteraction({
            geometryType: this.geometryType,
            style,
            whenStarted: oFeature => {
                this.drawFeature = new MeasureFeature(this.map, {oFeature, style},
                    this.geometryName + ' (' + this.defaultLabel + ')',
                    this.defaultLabel,
                );
            },
            whenEnded: oFeature => {
                let newFeature = new MeasureFeature(this.map, {geometry: oFeature.getGeometry(), style},
                    this.geometryName + ' (' + this.defaultLabel + ')',
                    this.defaultLabel,
                );
                layer.addFeature(newFeature);
                master.selectFeature(newFeature, false);
            }
        });

        let modify = this.map.modifyInteraction({
            style,
            source: layer.source,
            whenStarted: oFeatures => {
                if (oFeatures.length > 0) {
                    let fo = oFeatures[0]['_featureObj'];
                    if (fo)
                        master.selectFeature(fo, false);
                }
            }
        });

        this.map.setInteractions([
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
            'ZoomBox',
            draw,
            modify,
        ]);
    }

    stop() {
        console.log('MEASURE_STOP');
    }
}

class LineTool extends MeasureTool {
    geometryType = 'LineString';
    geometryName = this.__('modMeasureLine');
    cssSelector = '.modMeasureLine';
    defaultLabel = '{len}';
}

class PolygonTool extends MeasureTool {
    geometryType = 'Polygon';
    geometryName = this.__('modMeasurePolygon');
    cssSelector = '.modMeasurePolygon';
    defaultLabel = '{area}';
}

class CircleTool extends MeasureTool {
    geometryType = 'Circle';
    geometryName = this.__('modMeasureCircle');
    cssSelector = '.modMeasureCircle';
    defaultLabel = '{radius}';
}

class LineButton extends toolbar.ToolButton {
    className = 'modMeasureLineButton';
    tool = 'Tool.Measure.Line';

    get tooltip() {
        return this.__('modMeasureLineButton');
    }
}

class PolygonButton extends toolbar.ToolButton {
    className = 'modMeasurePolygonButton';
    tool = 'Tool.Measure.Polygon';

    get tooltip() {
        return this.__('modMeasurePolygonButton');
    }
}

class CircleButton extends toolbar.ToolButton {
    className = 'modMeasureCircleButton';
    tool = 'Tool.Measure.Circle';

    get tooltip() {
        return this.__('modMeasureCircleButton');
    }
}

class ClearButton extends toolbar.Button {
    className = 'modMeasureClearButton';

    get tooltip() {
        return this.__('modMeasureClearButton');
    }

    touched() {
        let master = this.app.controller(MASTER) as MeasureController;
        this.app.stopTool('Tool.Measure.*');
        master.clear();
        return this.update({
            toolbarItem: null,
        });
    }
}

class CancelButton extends toolbar.CancelButton {
    tool = 'Tool.Measure.*';
}

interface MeasureSidebarProps extends gws.types.ViewProps {
    controller: MeasureSidebarController;
    mapUpdateCount: number;
    measureSelectedFeature: MeasureFeature;
}

interface MeasureFeatureDetailsProps extends gws.types.ViewProps {
    controller: MeasureSidebarController;
    feature: MeasureFeature;
}

class MeasureFeatureDetailsToolbar extends gws.View<MeasureFeatureDetailsProps> {
    render() {

        let master = this.props.controller.app.controller(MASTER) as MeasureController;
        let f = this.props.feature;

        return <Row>
            <Cell flex/>
            <Cell>
                <gws.ui.IconButton
                    className="modMeasureFeatureDetailsSearchButton"
                    tooltip={this.__('modMeasureFeatureDetailsSearchButton')}
                    whenTouched={async () => await master.searchInFeature(f)}
                />
            </Cell>
            <Cell>
                <gws.ui.IconButton
                    className="modMeasureFeatureDetailsRemoveButton"
                    tooltip={this.__('modMeasureFeatureDetailsRemoveButton')}
                    whenTouched={() => master.removeFeature(f)}
                />
            </Cell>
            <Cell>
                <gws.ui.IconButton
                    className="modMeasureFeatureDetailsCloseButton"
                    tooltip={this.__('modMeasureFeatureDetailsCloseButton')}
                    whenTouched={() => master.unselectFeature(f)}
                />
            </Cell>
        </Row>;

    }
}

class MeasureFeatureDetails extends gws.View<MeasureFeatureDetailsProps> {
    render() {
        return <div className="modMeasureFeatureDetails">
            <div className="modMeasureFeatureDetailsBody">
                <gws.ui.TextArea
                    label={this.__('modMeasureLabelEdit')}
                    height={EDITOR_HEIGHT}
                    value={this.props.feature.labelTemplate}
                    whenChanged={value => this.props.feature.setLabelTemplate(value)}
                />
                <div className="uiHintText">
                    {this.__('modMeasureEditorHint')}
                </div>

            </div>
            <sidebar.SecondaryToolbar>
                <MeasureFeatureDetailsToolbar {...this.props} />
            </sidebar.SecondaryToolbar>
        </div>
    }
}

class MeasureSidebar extends gws.View<MeasureSidebarProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as MeasureController,
            layer = master.layer,
            selectedFeature = this.props.measureSelectedFeature;

        let head = <sidebar.TabHeader>
            <gws.ui.Title content={this.__('modMeasureTitle')}/>
        </sidebar.TabHeader>;

        if (!layer || layer.features.length === 0) {
            return <sidebar.Tab>
                {head}
                <sidebar.EmptyTab message={this.__('modMeasureNoFeaturesWarning')}/>
            </sidebar.Tab>;
        }

        return <sidebar.Tab>
            {head}

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={master}
                    features={layer.features}
                    isSelected={f => f === selectedFeature}

                    item={(f: MeasureFeature) => <gws.ui.Link
                        whenTouched={() => master.selectFeature(f, true)}
                        content={f.title}
                    />}

                    leftIcon={f => <gws.ui.IconButton
                        className="cmpFeatureZoomIcon"
                        whenTouched={() => this.props.controller.update({
                            marker: {
                                features: [f],
                                mode: 'zoom draw fade',
                            }
                        })}
                    />}
                />
            </sidebar.TabBody>

            {selectedFeature && <sidebar.TabFooter>
                <MeasureFeatureDetails {...this.props} feature={selectedFeature}/>
            </sidebar.TabFooter>}


        </sidebar.Tab>
    }
}

class MeasureSidebarController extends gws.Controller implements gws.types.ISidebarItem {

    get iconClass() {
        return 'modMeasureSidebarIcon'
    }

    get tooltip() {
        return this.__('modMeasureSidebarTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(MeasureSidebar, ['mapUpdateCount', 'measureSelectedFeature']),
            {map: this.map}
        );
    }

}

class MeasureController extends gws.Controller {
    uid = MASTER;
    layer: MeasureLayer;

    getOrCreateLayer(): MeasureLayer {
        if (!this.layer)
            this.layer = this.map.addServiceLayer(new MeasureLayer(this.map, {
                uid: '_measure',
            }));
        return this.layer;
    }

    async init() {

        await this.app.addTool('Tool.Measure.Line', this.app.createController(LineTool, this));
        await this.app.addTool('Tool.Measure.Polygon', this.app.createController(PolygonTool, this));
        await this.app.addTool('Tool.Measure.Circle', this.app.createController(CircleTool, this));
    }

    clear() {
        if (this.layer)
            this.map.removeLayer(this.layer);
        this.layer = null;
    }

    selectFeature(f, highlight) {
        if (highlight)
            this.update({
                measureSelectedFeature: f,
                marker: {
                    features: [f],
                    mode: 'pan draw fade',
                }
            });
        else
            this.update({
                measureSelectedFeature: f,
            });
    }

    unselectFeature(f) {
        this.update({
            measureSelectedFeature: null
        });
    }

    removeFeature(f) {
        if (this.layer)
            this.layer.removeFeature(f);
        this.update({measureSelectedFeature: null})

    }

    async searchInFeature(f) {
        // @TODO merge with identifty

        let params = await this.map.searchParams('', f.geometry),
            res = await this.app.server.searchFindFeatures(params);

        if (res.error) {
            console.log('SEARCH_ERROR', res);
            return [];
        }

        let features = this.map.readFeatures(res.features);

        if (features.length) {
            this.update({
                marker: {
                    features,
                    mode: 'draw',
                },
                popupContent: <gws.components.feature.PopupList controller={this} features={features}/>
            });
        } else {
            this.update({
                marker: {
                    features: null,
                },
                popupContent: null
            });
        }
    }

}

export const tags = {
    [MASTER]: MeasureController,
    'Toolbar.Measure.Line': LineButton,
    'Toolbar.Measure.Polygon': PolygonButton,
    'Toolbar.Measure.Circle': CircleButton,
    'Toolbar.Measure.Clear': ClearButton,
    'Toolbar.Measure.Cancel': CancelButton,
    'Sidebar.Measure': MeasureSidebarController,

};
