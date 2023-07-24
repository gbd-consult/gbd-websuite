import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as toolbar from './toolbar';
import * as sidebar from './sidebar';
import * as storage from './storage';
import * as toolbox from './toolbox';

let {Form, Row, Cell} = gws.ui.Layout;

const STORAGE_CATEGORY = 'Select';
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

    // layerPtr: LensLayer;
    ixDraw: ol.interaction.Draw;
    drawState: string = '';
    styleName: string;

    title = this.__('modLens');

    async init() {
        this.styleName = this.app.style.get('.modLensFeature').name;
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

        let master = _master(this.props.controller);

        let hasSelection = !gws.tools.empty(this.props.selectFeatures);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modSelectSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {hasSelection
                    ? <gws.components.feature.List
                        controller={this.props.controller}
                        features={this.props.selectFeatures}

                        content={(f) => <gws.ui.Link
                            whenTouched={() => master.focusFeature(f)}
                            content={master.featureTitle(f)}
                        />}

                        withZoom

                        rightButton={f => <gws.components.list.Button
                            className="modSelectUnselectListButton"
                            whenTouched={() => master.removeFeature(f)}
                        />}
                    />
                    : <sidebar.EmptyTabBody>
                        {this.__('modSelectNoObjects')}

                    </sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    {master.setup.exportFormats.length > 0 && <sidebar.AuxButton
                        className="modSelectExportAuxButton"
                        tooltip={this.__('modSelectExportAuxButton')}
                        disabled={!hasSelection}
                        whenTouched={() => master.whenExportButtonTouched()}
                    />}

                    <Cell flex/>
                    {storage.auxButtons(master, {
                        category: STORAGE_CATEGORY,
                        hasData: hasSelection,
                        getData: name => ({features: this.props.selectFeatures.map(f => f.getProps())}),
                        dataReader: (name, data) => master.loadFeatures(data.features)
                    })}
                    <sidebar.AuxButton
                        className="modSelectClearAuxButton"
                        tooltip={this.__('modSelectClearAuxButton')}
                        disabled={!hasSelection}
                        whenTouched={() => master.clear()}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class SelectSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modSelectSidebarIcon';

    get tooltip() {
        return this.__('modSelectSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SelectSidebarView, StoreKeys)
        );
    }

}

class SelectToolbarButton extends toolbar.Button {
    iconClass = 'modSelectToolbarButton';
    tool = 'Tool.Select';

    get tooltip() {
        return this.__('modSelectToolbarButton');
    }

}

class SelectDrawToolbarButton extends toolbar.Button {
    iconClass = 'modSelectDrawToolbarButton';
    tool = 'Tool.Select.Draw';

    get tooltip() {
        return this.__('modSelectDrawToolbarButton');
    }

}

class Dialog extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            params = this.props.selectFormatDialogParams;

        if (!params)
            return null;

        let listItems = cc.setup.exportFormats.map(e => ({
            text: e.title,
            value: e.uid
        }));


        let close = () => cc.update({selectFormatDialogParams: null});

        let submit = async () => cc.whenExportFormatSelected();

        let update = val => cc.update({selectSelectedFormat: val})

        let buttons = [
            <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={submit}
                disabled={gws.tools.empty(this.props.selectSelectedFormat)}
            />,
            <gws.ui.Button className="cmpButtonFormCancel" whenTouched={close}/>
        ];

        let cls = 'modSelectFormatDialog';
        let title = this.__('modSelectFormatDialogTitle');

        return <gws.ui.Dialog className={cls} title={title} buttons={buttons} whenClosed={close}>
            <Form>
                <Row>
                    <Cell flex>
                        <gws.ui.List
                            value={this.props.selectSelectedFormat}
                            items={listItems}
                            whenChanged={update}/>
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}

class SelectController extends gws.Controller {
    uid = MASTER;
    layer: gws.types.IFeatureLayer;
    setup: gws.api.SelectProps;

    async init() {
        this.setup = this.app.actionSetup('select');
        this.update({
            selectFeatures: []
        });
        this.app.whenCalled('selectFeature', args => {
            this.addFeature(args.feature);
        });
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }


    async doSelect(geometry: ol.geom.Geometry, toggle: boolean) {
        let features = await this.map.searchForFeatures({geometry});

        if (gws.tools.empty(features))
            return;

        features.forEach(f => this.addFeature(f, toggle));
    }

    whenExportButtonTouched() {
        let features = this.getValue('selectFeatures');

        if (gws.tools.empty(features))
            return;

        if (!this.setup.exportFormats)
            return;

        if (this.setup.exportFormats.length === 1) {
            this.doExportWithFormat(features, this.setup.exportFormats[0].uid);
            return;
        }

        this.update({selectFormatDialogParams: {}})
    }

    whenExportFormatSelected() {
        this.update({selectFormatDialogParams: null})
        let features = this.getValue('selectFeatures');
        let formatUid = this.getValue('selectSelectedFormat')
        this.doExportWithFormat(features, formatUid)
    }

    async doExportWithFormat(features, formatUid) {
        let res = await this.app.server.selectExport({
            exportFormatUid: formatUid,
            featureUids: features.map(f => f.uid),
        }, {binary: true} );

        let a = document.createElement('a');
        a.href = window.URL.createObjectURL(new Blob([res.content], {type: res.mime}));
        a.download = 'export.zip';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(a.href);
        document.body.removeChild(a);
    }

    addFeature(feature, toggle = false) {

        if (!this.layer) {
            this.layer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
                uid: '_select',
                cssSelector: '.modSelectFeature',
            }));
        }

        if (!feature.oFeature) {
            let geometry = feature.geometry;
            feature.oFeature = new ol.Feature({geometry});
        }

        let f = this.findFeatureByUid(feature.uid);

        if (f) {
            if (toggle)
                this.layer.removeFeatures([f]);
        } else
            this.layer.addFeatures([feature]);

        this.update({
            selectFeatures: this.layer.features
        });

    }

    featureTitle(feature: gws.types.IFeature) {
        if (feature.elements.title)
            return feature.elements.title;
        if (feature.elements.category)
            return feature.elements.category;
        return "...";
    }

    focusFeature(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'draw',
            },
            infoboxContent: <gws.components.feature.InfoList controller={this} features={[f]}/>
        });
    }

    loadFeatures(fs) {
        this.clear();
        this.map.readFeatures(fs).forEach(f => this.addFeature(f));
    }

    removeFeature(f) {
        this.layer.removeFeatures([f]);
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

}

export const tags = {
    [MASTER]: SelectController,
    'Toolbar.Select': SelectToolbarButton,
    'Toolbar.Select.Draw': SelectDrawToolbarButton,
    'Sidebar.Select': SelectSidebar,
    'Tool.Select': SelectTool,
    'Tool.Select.Draw': SelectDrawTool,
};
