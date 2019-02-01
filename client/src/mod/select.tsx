import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

import * as sidebar from './common/sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

interface SelectViewProps extends gws.types.ViewProps {
    selectFeatures: Array<gws.types.IMapFeature>;
}

const SelectStoreKeys = [
    'selectFeatures',
];

const MASTER = 'Shared.Select';

class SelectTool extends gws.Controller implements gws.types.ITool {

    async run(evt: ol.MapBrowserPointerEvent) {
        let master = this.app.controller(MASTER) as SelectController;
        let extend = !evt.originalEvent['altKey'];
        master.select(evt.coordinate, extend);
    }

    start() {

        this.map.setInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => this.run(evt),
            }),
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
            'ZoomBox',
        ]);
    }

    stop() {
    }

}

class SelectSidebar extends gws.View<SelectViewProps> {
    render() {

        let master = this.props.controller.app.controller(MASTER) as SelectController;

        if (gws.tools.empty(this.props.selectFeatures)) {
            return <sidebar.EmptyTab>
                {this.__('modSelectNoObjects')}
            </sidebar.EmptyTab>

        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modSelectTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={master}
                    features={this.props.selectFeatures}

                    item={(f) => <gws.ui.Link
                        whenTouched={() => master.focusFeature(f)}
                        content={master.featureTitle(f)}
                    />}

                    leftIcon={f => <gws.ui.IconButton
                        className="cmpFeatureZoomIcon"
                        whenTouched={() => master.zoomFeature(f)}
                    />}

                    rightIcon={f => <gws.ui.IconButton
                        className="cmpFeatureUnselectIcon"
                        whenTouched={() => master.removeFeature(f)}
                    />}
                />

            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Cell flex/>
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class SelectSidebarController extends gws.Controller implements gws.types.ISidebarItem {
    get iconClass() {
        return 'modSelectSidebarIcon';
    }

    get tooltip() {
        return this.__('modSelectTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SelectSidebar, SelectStoreKeys)
        );
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
            popupContent: <gws.components.feature.PopupList controller={this} features={[f]}/>
        });

    }

    removeFeature(f) {
        this.layer.removeFeature(f);
        this.update({
            selectFeatures: this.layer.features
        });

    }

}

class SelectToolbarButton extends toolbar.Button {
    className = 'modSelectToolbarButton';
    tool = 'Tool.Select';

    get tooltip() {
        return this.__('modSelectToolbarButton');
    }

}

export const tags = {
    [MASTER]: SelectController,
    'Toolbar.Select': SelectToolbarButton,
    'Sidebar.Select': SelectSidebarController,
};
