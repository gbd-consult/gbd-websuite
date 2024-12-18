import * as React from 'react';

import * as gc from 'gc';
import * as sidebar from 'gc/elements/sidebar';
import * as components from 'gc/components';

let {Form, Row, Cell} = gc.ui.Layout;

interface ViewProps extends gc.types.ViewProps {
    controller: Controller;
    mapUpdateCount: number;
    mapEditScale: number;
    mapEditAngle: number;
    mapEditCenterX: number;
    mapEditCenterY: number;

}

class SidebarBody extends gc.View<ViewProps> {


    submit() {
        let map = this.props.controller.map;

        map.setViewState({
            center: [this.props.mapEditCenterX, this.props.mapEditCenterY],
            scale: this.props.mapEditScale,
            angle: this.props.mapEditAngle
        }, true)
    }

    mapInfoBlock() {
        let map = this.props.controller.map;

        let coord = n => map.formatCoordinate(Number(n) || 0);
        let ext = map.viewExtent.map(coord).join(', ');

        let res = map.resolutions,
            maxScale = Math.max(...res.map(gc.lib.res2scale)),
            minScale = Math.min(...res.map(gc.lib.res2scale));

        let bind = k => ({
            whenChanged: v => this.props.controller.update({[k]: v}),
            whenEntered: () => this.submit(),
        });


        return <Form>
            <Row>
                <Cell flex>
                    <Form tabular>
                        <gc.ui.TextInput
                            label={this.__('overviewProjection')}
                            value={map.projection.getCode()}
                            readOnly
                        />
                        <gc.ui.TextInput
                            label={this.__('overviewExtent')}
                            value={ext}
                            readOnly
                        />
                        <gc.ui.TextInput
                            label={this.__('overviewCenterX')}
                            value={coord(this.props.mapEditCenterX)}
                            {...bind('mapEditCenterX')}
                        />
                        <gc.ui.TextInput
                            label={this.__('overviewCenterY')}
                            value={coord(this.props.mapEditCenterY)}
                            {...bind('mapEditCenterY')}
                        />
                        <gc.ui.NumberInput
                            minValue={minScale}
                            maxValue={maxScale}
                            step={1000}
                            label={this.__('overviewScale')}
                            value={this.props.mapEditScale}
                            {...bind('mapEditScale')}
                        />
                        <gc.ui.NumberInput
                            minValue={0}
                            maxValue={359}
                            step={5}
                            withClear
                            label={this.__('overviewRotation')}
                            value={this.props.mapEditAngle}
                            {...bind('mapEditAngle')}
                        />
                    </Form>
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gc.ui.Button
                        primary
                        whenTouched={() => this.submit()}
                        label={this.__('overviewUpdateButton')}
                    />
                </Cell>
            </Row>
        </Form>;
    }

    render() {
        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gc.ui.Title content={this.__('overviewSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <components.SmallMap controller={this.props.controller}/>
                {this.mapInfoBlock()}
            </sidebar.TabBody>

        </sidebar.Tab>
    }
}

class Controller extends gc.Controller implements gc.types.ISidebarItem {
    iconClass = 'overviewSidebarIcon';

    async init() {
        this.app.whenChanged('mapUpdateCount', () => this.refresh());
        this.app.whenChanged('windowSize', () => this.refresh());
    }

    get tooltip() {
        return this.__('overviewSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarBody, [
                'mapUpdateCount',
                'mapEditScale',
                'mapEditAngle',
                'mapEditCenterX',
                'mapEditCenterY',
            ]),
        );
    }

    refresh() {
        let vs = this.map.viewState;

        this.update({
            'mapEditScale': vs.scale,
            'mapEditAngle': vs.angle,
            'mapEditCenterX': vs.centerX,
            'mapEditCenterY': vs.centerY,
        });
    }

}

gc.registerTags({
    'Sidebar.Overview': Controller
});

