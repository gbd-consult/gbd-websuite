import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

import * as sidebar from './common/sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Select';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as SelectController;

interface SelectViewProps extends gws.types.ViewProps {
    controller: SelectController;
    selectFeatures: Array<gws.types.IMapFeature>;
    selectDialogMode: string;
    selectSaveName: string;
    selectSaveNames: Array<string>;
}

const SelectStoreKeys = [
    'selectFeatures',
    'selectDialogMode',
    'selectSaveName',
    'selectSaveNames',
];

class SelectDialog extends gws.View<SelectViewProps> {

    render() {
        let mode = this.props.selectDialogMode;

        if (!mode)
            return null;

        let close = () => this.props.controller.update({selectDialogMode: null});
        let update = v => this.props.controller.update({selectSaveName: v});

        let title, submit, control;

        if (mode === 'save') {
            title = this.__('modSelectSaveDialogTitle');
            submit = () => this.props.controller.saveSelection();
            control = <gws.ui.TextInput
                value={this.props.selectSaveName}
                whenChanged={update}
                whenEntered={submit}
            />;
        }

        if (mode === 'load') {
            title = this.__('modSelectLoadDialogTitle');
            submit = () => this.props.controller.loadSelection();
            control = <gws.ui.Select
                value={this.props.selectSaveName}
                items={this.props.selectSaveNames.map(s => ({
                    text: s,
                    value: s

                }))}
                whenChanged={update}
            />;
        }

        return <gws.ui.Dialog className="modSelectDialog" title={title} whenClosed={close}>
            <Form>
                <Row>
                    <Cell flex>{control}</Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton className="cmpButtonFormOk" whenTouched={submit}/>
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton className="cmpButtonFormCancel" whenTouched={close}/>
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}

class SelectTool extends gws.Tool {

    async run(evt: ol.MapBrowserPointerEvent) {
        let extend = !evt.originalEvent['altKey'];
        await _master(this).select(evt.coordinate, extend);
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

class SelectSidebarView extends gws.View<SelectViewProps> {
    render() {

        let master = _master(this.props.controller);

        let hasSelection = !gws.tools.empty(this.props.selectFeatures);
        let hasSaveNames = !gws.tools.empty(this.props.selectSaveNames);

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
                    <Cell flex/>
                    <sidebar.AuxButton
                        className="modSelectSaveAuxButton"
                        tooltip={this.__('modSelectSaveAuxButton')}
                        disabled={!hasSelection}
                        whenTouched={() => master.update({selectDialogMode: 'save'})}
                    />
                    <sidebar.AuxButton
                        className="modSelectLoadAuxButton"
                        tooltip={this.__('modSelectLoadAuxButton')}
                        disabled={!hasSaveNames}
                        whenTouched={() => master.update({selectDialogMode: 'load'})}
                    />
                    <sidebar.AuxButton
                        className="modSelectClearAuxButton"
                        tooltip={this.__('modSelectClearAuxButton')}
                        disabled={!hasSelection}
                        whenTouched={() => master.clearSelection()}
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
            this.connect(SelectSidebarView, SelectStoreKeys)
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

class SelectController extends gws.Controller {
    uid = MASTER;
    layer: gws.types.IMapFeatureLayer;

    async init() {
        await this.getSaveNames();
        this.app.whenCalled('selectFeature', args => {
            this.addFeature(args.feature);
        });
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(SelectDialog, SelectStoreKeys));
    }

    async select(center: ol.Coordinate, extend: boolean) {
        let features = await this.map.searchForFeatures({geometry: new ol.geom.Point(center)});

        if (gws.tools.empty(features))
            return;

        this.addFeature(features[0], extend);
    }

    addFeature(feature, extend = true) {

        if (!this.layer) {
            this.layer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
                uid: '_select',
                style: this.map.getStyleFromSelector('.modSelectFeature'),
            }));
        }

        if (!extend)
            this.layer.clear();

        if (!feature.oFeature) {
            let geometry = feature.geometry;
            feature.oFeature = new ol.Feature({geometry});
        }

        console.log(feature)

        this.layer.addFeatures([feature]);
        this.update({
            selectFeatures: this.layer.features
        });

    }

    featureTitle(feature: gws.types.IMapFeature) {
        if (feature.props.title)
            return feature.props.title;
        if (feature.props.category)
            return feature.props.category;
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

    async saveSelection() {
        let res = await this.app.server.selectSaveFeatures({
            projectUid: this.app.project.uid,
            name: this.getValue('selectSaveName'),
            features: this.getValue('selectFeatures').map(f => f.props)
        });

        if (res) {
            this.update({
                selectSaveNames: res.names
            });
        }

        this.update({
            selectDialogMode: null,
        });
    }

    async loadSelection() {
        let res = await this.app.server.selectLoadFeatures({
            projectUid: this.app.project.uid,
            name: this.getValue('selectSaveName'),
        });

        if (res.features) {
            this.clearSelection();
            this.map.readFeatures(res.features).forEach(f => this.addFeature(f));
        }

        this.update({
            selectDialogMode: null,
        });
    }

    async getSaveNames() {
        let res = await this.app.server.selectGetSaveNames({
            projectUid: this.app.project.uid,
        });
        this.update({
            selectSaveNames: res.names || []
        })

    }

    removeFeature(f) {
        this.layer.removeFeature(f);
        this.update({
            selectFeatures: this.layer.features
        });
    }

    clearSelection() {
        if (this.layer)
            this.map.removeLayer(this.layer);
        this.layer = null;

        this.update({
            selectFeatures: null
        });

    }

}

export const tags = {
    [MASTER]: SelectController,
    'Toolbar.Select': SelectToolbarButton,
    'Sidebar.Select': SelectSidebar,
    'Tool.Select': SelectTool,
};
