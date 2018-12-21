import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './toolbar';
import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Measure';

class MeasureLayer extends gws.map.layer.FeatureLayer {
}

interface MeasureFeatureFormData {
    x?: string
    y?: string
    w?: string
    h?: string
    radius?: string
    labelTemplate: string
}

interface MeasureFeatureArgs extends gws.types.IMapFeatureArgs {
    labelTemplate: string;
    selectedStyle: gws.types.IMapStyle;
    shapeType: string;
}

class MeasureFeature extends gws.map.Feature {
    app: gws.types.IApplication;
    labelTemplate: string;
    selected: boolean;
    selectedStyle: gws.types.IMapStyle;
    shapeType: string;
    cc: any;

    get formData(): MeasureFeatureFormData {
        let dims = this.computeDimensions(this.geometry, this.map.projection);
        return {
            x: this.formatCoordinate(dims.x),
            y: this.formatCoordinate(dims.y),
            radius: this.formatEditLength(dims.radius),
            w: this.formatEditLength(dims.w),
            h: this.formatEditLength(dims.h),
            labelTemplate: this.labelTemplate
        }
    }

    constructor(app: gws.types.IApplication, args: MeasureFeatureArgs) {
        super(app.map, args);
        this.app = app;
        this.selected = false;
        this.labelTemplate = args.labelTemplate;
        this.selectedStyle = args.selectedStyle;
        this.shapeType = args.shapeType;

        this.oFeature.setStyle((oFeature, r) => {
            let s = this.selected ? this.selectedStyle : this.style;
            return s.apply(oFeature.getGeometry(), this.label, r);
        });
        this.geometry.on('change', e => this.onChange(e));
        this.redraw();
    }

    onChange(e) {
        this.redraw();
    }

    redraw() {
        let master = this.app.controller(MASTER) as MeasureController;
        let dims = this.computeDimensions(this.geometry, this.map.projection);
        this.setLabel(this.format(this.labelTemplate, dims));
        master.featureUpdated(this);
    }

    protected format(text, dims) {
        return (text || '').replace(/{(\w+)}/g, ($0, key) => this.formatDimensionElement(dims, key));
    }

    protected computeDimensions(geom, projection) {

        let gt = geom.getType();

        if (gt === 'Point') {
            let g = geom as ol.geom.Point;
            let c = g.getCoordinates();
            return {
                x: c[0],
                y: c[1]
            }
        }

        if (gt === 'LineString') {
            return {
                len: ol.Sphere.getLength(geom, {projection})
            }
        }

        if (gt === 'Polygon') {
            let g = geom as ol.geom.Polygon;
            let r = g.getLinearRing(0);
            let c: any = r.getCoordinates();
            let d = {
                len: ol.Sphere.getLength(r, {projection}),
                area: ol.Sphere.getArea(g, {projection}),
            };

            if (this.shapeType === 'Box') {
                // NB: x,y = top left
                return {
                    ...d,
                    x: c[0][0],
                    y: c[3][1],
                    w: Math.abs(c[1][0] - c[0][0]),
                    h: Math.abs(c[2][1] - c[1][1]),
                }
            }
            return d;
        }

        if (gt === 'Circle') {
            let g = geom as ol.geom.Circle;
            let p = ol.geom.Polygon.fromCircle(g, 64, 0);
            let c = g.getCenter();

            return {
                ...this.computeDimensions(p, projection),
                x: c[0],
                y: c[1],
                radius: g.getRadius(),
            }
        }

        return {};
    }

    protected formatDimensionElement(dims: object, key: string) {
        let n = dims[key];
        if (!n)
            return '';
        switch (key) {
            case 'area':
                return this.formatArea(n);
            case 'len':
            case 'radius':
            case 'w':
            case 'h':
                return this.formatLength(n);
            case 'x':
            case 'y':
                return this.formatCoordinate(n);

        }
    }

    protected formatLength(n) {
        if (!n || n < 0.01)
            return '';
        if (n >= 1e3)
            return (n / 1e3).toFixed(2) + ' km';
        if (n > 1)
            return n.toFixed(0) + ' m';
        return n.toFixed(2) + ' m';
    }

    protected formatArea(n) {
        let sq = '\u00b2';

        if (!n || n < 0.01)
            return '';
        if (n >= 1e5)
            return (n / 1e6).toFixed(2) + ' km' + sq;
        if (n > 1)
            return n.toFixed(0) + ' m' + sq;
        return n.toFixed(2) + ' m' + sq;
    }

    protected formatCoordinate(n) {
        return this.map.formatCoordinate(Number(n) || 0);
    }

    protected formatEditLength(n) {
        return (Number(n) || 0).toFixed(0)
    }

    updateFromForm(ff: MeasureFeatureFormData) {
        this.labelTemplate = ff.labelTemplate;

        let t = this.shapeType;

        if (t === 'Point') {
            let g = this.geometry as ol.geom.Point;
            g.setCoordinates([
                Number(ff.x) || 0,
                Number(ff.y) || 0,

            ]);
        }

        if (t === 'Circle') {
            let g = this.geometry as ol.geom.Circle;

            g.setCenter([
                Number(ff.x) || 0,
                Number(ff.y) || 0,

            ]);
            g.setRadius(Number(ff.radius) || 1)
        }

        if (t === 'Box') {
            // NB: x,y = top left
            let g = this.geometry as ol.geom.Polygon;
            let x = Number(ff.x) || 0,
                y = Number(ff.y) || 0,
                w = Number(ff.w) || 100,
                h = Number(ff.h) || 100;
            let coords: any = [
                [x, y - h],
                [x + w, y - h],
                [x + w, y],
                [x, y],
                [x, y - h],
            ];
            g.setCoordinates([coords])
        }

        this.geometry.changed();
    }

}

abstract class MeasureTool extends gws.Controller implements gws.types.ITool {
    shapeType: string;
    cssSelector: string;
    defaultLabel: string;

    drawFeature: MeasureFeature;
    newFeature: MeasureFeature;

    start() {
        let master = this.app.controller(MASTER) as MeasureController,
            layer = master.getOrCreateLayer(),
            style = this.map.getStyleFromSelector(this.cssSelector),
            selectedStyle = this.map.getStyleFromSelector(this.cssSelector + 'Selected');

        console.log('MEASURE_START');

        this.newFeature = null;

        let startDraw = oFeature => {

            // if (this.newFeature)
            //     layer.removeFeature(this.newFeature);

            this.drawFeature = new MeasureFeature(master.app, {
                oFeature,
                shapeType: this.shapeType,
                style,
                labelTemplate: this.defaultLabel,
                selectedStyle
            });
        };

        let endDraw = oFeature => {

            this.newFeature = new MeasureFeature(master.app, {
                geometry: oFeature.getGeometry(),
                shapeType: this.shapeType,
                style,
                labelTemplate: this.defaultLabel,
                selectedStyle
            });
            layer.addFeature(this.newFeature);
            master.selectFeature(this.newFeature, false);

            this.app.stopTool(this.getValue('activeTool'))
            this.update({
                toolbarItem: null,
            })

        };

        let startModify = oFeatures => {
            if (oFeatures.length > 0) {
                let fo = oFeatures[0]['_featureObj'];
                if (fo)
                    master.selectFeature(fo, false);
            }
        };

        let drawOpts: gws.types.DrawInteractionOptions = {
            geometryType: this.shapeType,
            style: selectedStyle,
            whenStarted: startDraw,
            whenEnded: endDraw
        };

        if (this.shapeType === 'Box') {
            drawOpts.geometryType = 'Circle';
            drawOpts.geometryFunction = ol.interaction.Draw.createBox();
        }

        let draw = this.map.drawInteraction(drawOpts);

        let modify = this.map.modifyInteraction({
            style: selectedStyle,
            source: layer.source,
            whenStarted: startModify,
        });

        this.map.setInteractions([
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
            'ZoomBox',
            draw,
            //modify,
        ]);
    }

    stop() {
        console.log('MEASURE_STOP');
    }
}

class PointTool extends MeasureTool {
    shapeType = 'Point';
    cssSelector = '.modMeasurePoint';
    defaultLabel = '{x}, {y}';
}

class LineTool extends MeasureTool {
    shapeType = 'LineString';
    cssSelector = '.modMeasureLine';
    defaultLabel = '{len}';
}

class PolygonTool extends MeasureTool {
    shapeType = 'Polygon';
    cssSelector = '.modMeasurePolygon';
    defaultLabel = '{area}';
}

class CircleTool extends MeasureTool {
    shapeType = 'Circle';
    cssSelector = '.modMeasureCircle';
    defaultLabel = '{radius}';
}

class BoxTool extends MeasureTool {
    shapeType = 'Box';
    cssSelector = '.modMeasureBox';
    defaultLabel = '{w} x {h}';
}

class PointButton extends toolbar.ToolButton {
    className = 'modMeasurePointButton';
    tool = 'Tool.Measure.Point';

    get tooltip() {
        return this.__('modMeasurePointButton');
    }
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

class BoxButton extends toolbar.ToolButton {
    className = 'modMeasureBoxButton';
    tool = 'Tool.Measure.Box';

    get tooltip() {
        return this.__('modMeasureBoxButton');
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
    measureFeatureForm: MeasureFeatureFormData;
}

class MeasureFeatureDetailsToolbar extends gws.View<MeasureSidebarProps> {
    render() {

        let master = this.props.controller.app.controller(MASTER) as MeasureController;
        let f = this.props.measureSelectedFeature;

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
                    whenTouched={() => master.unselectFeature()}
                />
            </Cell>
        </Row>;

    }
}

class MeasureFeatureDetails extends gws.View<MeasureSidebarProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as MeasureController;
        let f = this.props.measureSelectedFeature;
        let ff = this.props.measureFeatureForm;

        let changed = (key, val) => master.update({
            measureFeatureForm: {
                ...ff,
                [key]: val
            }
        });

        let submit = () => {
            f.updateFromForm(this.props.measureFeatureForm);
        };

        let data = [];
        let t = f.shapeType;

        if (t === 'Point') {
            data.push({
                name: 'x',
                title: this.__('modMeasureX'),
                value: ff.x,
                editable: true
            });
            data.push({
                name: 'y',
                title: this.__('modMeasureY'),
                value: ff.y,
                editable: true
            });
        }

        if (t === 'Box') {
            data.push({
                name: 'x',
                title: this.__('modMeasureX'),
                value: ff.x,
                editable: true
            });
            data.push({
                name: 'y',
                title: this.__('modMeasureY'),
                value: ff.y,
                editable: true
            });
            data.push({
                name: 'w',
                title: this.__('modMeasureWidth'),
                value: ff.w,
                editable: true
            });
            data.push({
                name: 'h',
                title: this.__('modMeasureHeight'),
                value: ff.h,
                editable: true
            });
        }

        if (t === 'Circle') {
            data.push({
                name: 'x',
                title: this.__('modMeasureX'),
                value: ff.x,
                editable: true
            });
            data.push({
                name: 'y',
                title: this.__('modMeasureY'),
                value: ff.y,
                editable: true
            });
            data.push({
                name: 'radius',
                title: this.__('modMeasureRaidus'),
                value: ff.radius,
                editable: true
            });
        }

        data.push({
            name: 'labelTemplate',
            title: this.__('modMeasureLabelEdit'),
            value: ff['labelTemplate'],
            type: 'text',
            editable: true
        })

        let form = <Form>
            <Row>
                <Cell flex>
                    <gws.components.sheet.Editor
                        data={data}
                        whenChanged={changed}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.TextButton primary whenTouched={submit}>
                        {this.__('modMeasureUpdateButton')}
                    </gws.ui.TextButton>
                </Cell>
            </Row>
        </Form>;

        return <div className="modMeasureFeatureDetails">
            <div className="modMeasureFeatureDetailsBody">
                {form}
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
                        content={f.label}
                    />}

                    leftIcon={f => <gws.ui.IconButton
                        className="cmpFeatureZoomIcon"
                        whenTouched={() => master.update({
                            marker: {
                                features: [f],
                                mode: 'zoom draw fade',
                            }
                        })}
                    />}
                />
            </sidebar.TabBody>

            {selectedFeature && <sidebar.TabFooter>
                <MeasureFeatureDetails {...this.props} />
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
            this.connect(MeasureSidebar, ['mapUpdateCount', 'measureSelectedFeature', 'measureFeatureForm']),
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

        await this.app.addTool('Tool.Measure.Point', this.app.createController(PointTool, this));
        await this.app.addTool('Tool.Measure.Line', this.app.createController(LineTool, this));
        await this.app.addTool('Tool.Measure.Polygon', this.app.createController(PolygonTool, this));
        await this.app.addTool('Tool.Measure.Circle', this.app.createController(CircleTool, this));
        await this.app.addTool('Tool.Measure.Box', this.app.createController(BoxTool, this));
    }

    clear() {
        if (this.layer)
            this.map.removeLayer(this.layer);
        this.layer = null;
    }

    selectFeature(f: MeasureFeature, highlight) {
        this.layer.features.forEach(f => (f as MeasureFeature).selected = false);
        f.selected = true;

        this.update({
            measureSelectedFeature: f,
            measureFeatureForm: f.formData,
        });

        if (highlight) {
            this.update({
                marker: {
                    features: [f],
                    mode: 'pan',
                }
            });
            f.oFeature.changed();
        } else {
            f.redraw();
        }
    }

    unselectFeature() {
        this.layer.features.forEach(f => (f as MeasureFeature).selected = false);
        this.update({
            measureSelectedFeature: null,
            measureFeatureForm: {},
        });
    }

    featureUpdated(f) {
        let sel = this.getValue('measureSelectedFeature');
        if (f === sel)
            this.update({
                measureFeatureForm: f.formData,
            });
    }

    removeFeature(f) {
        if (this.layer)
            this.layer.removeFeature(f);
        this.unselectFeature();

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
    'Toolbar.Measure.Point': PointButton,
    'Toolbar.Measure.Line': LineButton,
    'Toolbar.Measure.Polygon': PolygonButton,
    'Toolbar.Measure.Circle': CircleButton,
    'Toolbar.Measure.Box': BoxButton,
    'Toolbar.Measure.Clear': ClearButton,
    'Toolbar.Measure.Cancel': CancelButton,
    'Sidebar.Measure': MeasureSidebarController,

};
