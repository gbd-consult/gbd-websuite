import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

import * as sidebar from './common/sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Select';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as SelectController;


interface SelectViewProps extends gws.types.ViewProps {
    selectFeatures: Array<gws.types.IMapFeature>;
}

const SelectStoreKeys = [
    'selectFeatures',
];

class SelectTool extends gws.Controller implements gws.types.ITool {

    async run(evt: ol.MapBrowserPointerEvent) {
        let extend = !evt.originalEvent['altKey'];
        await _master(this).select(evt.coordinate, extend);
    }

    start() {
        this.map.setExtraInteractions([
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

        if (gws.tools.empty(this.props.selectFeatures)) {
            return <sidebar.EmptyTab>
                {this.__('modSelectNoObjects')}
            </sidebar.EmptyTab>
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modSelectSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
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

            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
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
        await this.app.addTool('Tool.Select', this.app.createController(SelectTool, this));

        this.whenChanged('selectAddFeature', f => {
            if (f) {
                this.addFeature(f);
                this.update({selectAddFeature: null})

            }
        });
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

        this.layer.addFeatures([feature]);
        this.update({
            selectFeatures: this.layer.features
        });

    }

    featureTitle(f) {
        if (f.props.title)
            return f.props.title;
        if (f.props.category)
            return f.props.category;
        return "...";
    }

    zoomFeature(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade',
            }
        })
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

    removeFeature(f) {
        this.layer.removeFeature(f);
        this.update({
            selectFeatures: this.layer.features
        });

    }

}

export const tags = {
    [MASTER]: SelectController,
    'Toolbar.Select': SelectToolbarButton,
    'Sidebar.Select': SelectSidebar,
};
