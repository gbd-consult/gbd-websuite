import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from '../gws';

import * as toolbar from './toolbar';
import * as sidebar from './sidebar';
import * as storage from './storage';

let {Form, Row, Cell} = gws.ui.Layout;

const STORAGE_CATEGORY = 'Select';
const MASTER = 'Shared.Select';


let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as SelectController;

interface SelectViewProps extends gws.types.ViewProps {
    controller: SelectController;
    selectFeatures: Array<gws.types.IFeature>;
}

const SelectStoreKeys = [
    'selectFeatures',
];


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

        let hasSelection = !gws.lib.empty(this.props.selectFeatures);

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
    layer: gws.types.IFeatureLayer;

    async init() {
        this.update({
            selectFeatures: []
        });
        this.app.whenCalled('selectFeature', args => {
            this.addFeature(args.feature);
        });
    }

    async select(center: ol.Coordinate, extend: boolean) {
        let features = await this.map.searchForFeatures({geometry: new ol.geom.Point(center)});

        if (gws.lib.empty(features))
            return;

        this.addFeature(features[0], extend);
    }

    addFeature(feature, extend = true) {

        if (!this.layer) {
            this.layer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
                uid: '_select',
                style: '.modSelectFeature',
            }));
        }

        if (!extend)
            this.layer.clear();

        if (!feature.oFeature) {
            let geometry = feature.geometry;
            feature.oFeature = new ol.Feature({geometry});
        }

        console.log(feature);

        let f = this.findFeatureByUid(feature.uid);

        if (f)
            this.layer.removeFeature(f);
        else
            this.layer.addFeatures([feature]);

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
            infoboxContent: <gws.components.feature.InfoList controller={this} features={[f]}/>
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

}

gws.registerTags({
    [MASTER]: SelectController,
    'Toolbar.Select': SelectToolbarButton,
    'Sidebar.Select': SelectSidebar,
    'Tool.Select': SelectTool,
});
