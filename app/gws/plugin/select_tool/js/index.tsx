import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as components from 'gws/components';
import * as storage from 'gws/elements/storage';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Select';


let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as SelectController;

interface ViewProps extends gws.types.ViewProps {
    controller: SelectController;
    selectFeatures: Array<gws.types.IFeature>;
    selectShapeType: string;
    selectSelectedFormat: string;
    selectFormatDialogParams: object;
}

const StoreKeys = [
    'selectFeatures',
    'selectShapeType',
    'selectSelectedFormat',
    'selectFormatDialogParams',
];


class SelectTool extends gws.Tool {

    async run(evt: ol.MapBrowserPointerEvent) {
        let toggle = !evt.originalEvent['altKey'];
        await _master(this).doSelect(new ol.geom.Point(evt.coordinate), toggle);
    }

    start() {
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Select'});
        this.map.prependInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => this.run(evt),
            }),
        ]);
    }

    stop() {
    }

}

export class SelectDrawTool extends gws.Tool {

    ixDraw: ol.interaction.Draw;
    drawState: string = '';
    styleName: string = '.selectFeature';

    async init() {
        this.update({selectShapeType: 'Polygon'});
    }

    start() {
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Select'});

        this.reset();

        let drawFeature;

        this.ixDraw = this.map.drawInteraction({
            shapeType: this.getValue('selectShapeType'),
            style: this.styleName,
            whenStarted: (oFeatures) => {
                drawFeature = oFeatures[0];
                this.drawState = 'drawing';
            },
            whenEnded: () => {
                if (this.drawState === 'drawing') {
                    this.run(drawFeature.getGeometry());
                }
                this.drawState = '';
            },
        });

        this.map.appendInteractions([this.ixDraw]);
    }

    stop() {
        this.reset();
    }

    async run(geom) {
        await _master(this).doSelect(geom, false);
    }

    setShapeType(st) {
        this.drawCancel();
        this.update({selectShapeType: st});
        this.start();
    }

    drawCancel() {
        if (this.drawState === 'drawing') {
            this.drawState = 'cancel';
            this.ixDraw.finishDrawing();
        }
    }

    reset() {
        this.map.resetInteractions();
        this.ixDraw = null;
    }
}

class SelectSidebarView extends gws.View<ViewProps> {
    render() {

        let cc = _master(this.props.controller);

        let hasSelection = !gws.lib.isEmpty(this.props.selectFeatures);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('selectSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {hasSelection
                    ? <components.feature.List
                        controller={this.props.controller}
                        features={this.props.selectFeatures}

                        content={(f) => <gws.ui.Link
                            whenTouched={() => cc.focusFeature(f)}
                            content={cc.featureTitle(f)}
                        />}

                        withZoom

                        rightButton={f => <components.list.Button
                            className="selectUnselectListButton"
                            whenTouched={() => cc.removeFeature(f)}
                        />}
                    />
                    : <sidebar.EmptyTabBody>
                        {this.__('selectNoObjects')}

                    </sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <storage.AuxButtons
                        controller={cc}
                        actionName="selectStorage"
                        hasData={hasSelection}
                        getData={() => cc.storageGetData()}
                        loadData={(data) => cc.storageLoadData(data)}
                    />
                    <sidebar.AuxButton
                        className="selectClearAuxButton"
                        tooltip={this.__('selectClearAuxButton')}
                        disabled={!hasSelection}
                        whenTouched={() => cc.clear()}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class SelectSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'selectSidebarIcon';

    get tooltip() {
        return this.__('selectSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SelectSidebarView, StoreKeys)
        );
    }

}

class SelectToolbarButton extends toolbar.Button {
    iconClass = 'selectToolbarButton';
    tool = 'Tool.Select';

    get tooltip() {
        return this.__('selectToolbarButton');
    }

}

class SelectDrawToolbarButton extends toolbar.Button {
    iconClass = 'selectDrawToolbarButton';
    tool = 'Tool.Select.Draw';

    get tooltip() {
        return this.__('selectDrawToolbarButton');
    }

}

class SelectController extends gws.Controller {
    uid = MASTER;
    layer: gws.types.IFeatureLayer;
    tolerance: string = '';

    async init() {
        this.update({
            selectFeatures: []
        });
        this.app.whenCalled('selectFeature', args => {
            this.addFeature(args.feature);
        });

        let setup = this.app.actionProps('select') as gws.api.plugin.select_tool.action.Props;
        if (setup) {
            this.tolerance = setup.tolerance || '';
            this.updateObject('storageState', {
                selectStorage: setup.storage ? setup.storage.state : null,
            })
        }
    }

    async doSelect(geometry: ol.geom.Geometry, toggle: boolean) {
        let features = await this.map.searchForFeatures({geometry, tolerance: this.tolerance});

        if (gws.lib.isEmpty(features))
            return;

        features.forEach(f => this.addFeature(f, toggle));
    }

    addFeature(feature: gws.types.IFeature, toggle = false) {

        if (!this.layer) {
            this.layer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
                uid: '_select',
                cssSelector: '.selectFeature',
            }));
        }

        if (!feature.oFeature) {
            let geometry = feature.geometry;
            feature.oFeature = new ol.Feature({geometry});
        }

        let f = this.findFeatureByUid(feature.uid);

        if (f) {
            if (toggle)
                this.layer.removeFeature(f);
        } else {
            this.layer.addFeatures([feature]);
        }

        this.update({
            selectFeatures: this.layer.features
        });
    }

    featureTitle(feature: gws.types.IFeature) {
        if (feature.views.title)
            return feature.views.title;
        if (feature.views.category)
            return feature.views.category;
        return "...";
    }

    focusFeature(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'draw',
            },
            infoboxContent: <components.feature.InfoList controller={this} features={[f]}/>
        });
    }

    loadFeatures(fs) {
        this.clear();
        this.map.readFeatures(fs).forEach(f => this.addFeature(f));
    }

    removeFeature(f) {
        this.layer.removeFeature(f);
        this.update({
            selectFeatures: this.layer.features
        });
    }

    findFeatureByUid(uid) {
        for (let f of this.layer.features)
            if (f.uid === uid)
                return f;
    }

    clear() {
        if (this.layer)
            this.map.removeLayer(this.layer);
        this.layer = null;

        this.update({
            selectFeatures: []
        });

    }

    storageGetData() {
        let fs = (this.getValue("selectFeatures") || []) as Array<gws.types.IFeature>;
        return {
            features: fs.map(f => f.getProps())
        }
    }

    storageLoadData(data) {
        let fs = data.features || [];
        this.clear();
        this.map.readFeatures(fs).forEach(f => this.addFeature(f));
    }

}

gws.registerTags({
    [MASTER]: SelectController,
    'Toolbar.Select': SelectToolbarButton,
    'Toolbar.Select.Draw': SelectDrawToolbarButton,
    'Sidebar.Select': SelectSidebar,
    'Tool.Select': SelectTool,
    'Tool.Select.Draw': SelectDrawTool,
});
