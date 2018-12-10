import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './sidebar';

class ModifyTool extends gws.Controller implements gws.types.ITool {
    parent: Controller;
    normStyle: gws.types.IMapStyle;

    start() {

        console.log('EditorModify_START');

        let layer = this.parent.layer;

        this.normStyle = layer.style;
        layer.style = layer.editStyle;
        layer.reset();

        let modify = this.map.modifyInteraction({
            style: layer.editStyle,
            source: layer.source,
            whenStarted: fs => {
                console.log('MODIFY STARTED', fs.length)
            },
            whenEnded: async fs => {
                console.log('MODIFY ENDED', fs.length)
                await this.parent.updateRaw(fs);
            },

        });
        let snap = new ol.interaction.Snap({source: layer.source});

        this.map.setInteractions([
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
            modify,
            snap,
        ]);

    }

    stop() {
        console.log('EditorModify_STOP');

        let layer = this.parent.app.store.getValue('editorLayer');

        if (layer) {
            layer.style = this.normStyle;
            layer.reset();
        }
    }
}

class DrawTool extends gws.Controller implements gws.types.ITool {
    parent: Controller;

    start() {

        // console.log('EditorDraw_START');
        //
        // let layer = this.parent.layer;
        //
        // let draw = this.map.drawInteraction({
        //     geometryType: layer.meta.geometry.type,
        //     style: layer.editStyle,
        //     whenStarted: feature => {
        //     },
        //     whenEnded: async f => {
        //         await this.parent.addFeature(f);
        //     }
        // });
        //
        // let snap = new ol.interaction.Snap({source: layer.source});
        //
        // this.map.setInteractions([
        //     draw,
        //     snap,
        //     'DragPan',
        //     'MouseWheelZoom',
        //     'PinchZoom',
        // ]);

    }

    stop() {
        console.log('EditorDraw_STOP');
    }
}

class PointerTool extends gws.Controller implements gws.types.ITool {
    parent: Controller;

    start() {

        console.log('EditorPoint_START');

        let layer = this.parent.app.store.getValue('editorLayer');

        let ptr = this.map.pointerInteraction({
            whenTouched: evt => this.parent.selectFromEvent(evt),
            hover: 'shift',
        })

        this.map.setInteractions([
            ptr,
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
        ]);

    }

    stop() {
        console.log('EditorPoint_STOP');
    }
}

interface BodyProps extends gws.types.ViewProps {
    controller: Controller;
    editorLayer: gws.types.IMapFeatureLayer;
    editorFeature: gws.types.IMapFeature;
    editorData: Array<gws.components.sheet.Attribute>;
}

class FeatureList extends gws.View<BodyProps> {
    render() {
        let {Row, Cell} = gws.ui.Layout;
        let cc = this.props.controller;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.props.editorLayer.title}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {/*<gws.components.feature.List*/}
                    {/*features={this.props.editorLayer.features}*/}
                    {/*icon="zoom"*/}
                    {/*whenTitleTouched={f => cc.showDetails(f)}*/}
                    {/*whenIconTouched={f => cc.highlight(f)}*/}
                {/*/>*/}
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            className="modEditorPointButton"
                            tooltip={this.__('modEditorPointerTool')}
                            whenTouched={() => cc.startTool('EditorPointerTool')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="modEditorModifyButton"
                            tooltip={this.__('modEditorModifyTool')}
                            whenTouched={() => cc.startTool('EditorModifyTool')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="modEditorDrawButton"
                            tooltip={this.__('modEditorDrawTool')}
                            whenTouched={() => cc.startTool('EditorDrawTool')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="modEditorEndButton"
                            tooltip={this.__('modEditorEndButton')}
                            whenTouched={() => cc.endEditing()}
                        />
                    </Cell>
                </Row>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class FeatureDetails extends gws.View<BodyProps> {
    render() {
        let {Row, Cell} = gws.ui.Layout;
        let cc = this.props.controller;
        let data = this.props.editorData;

        let changed = (k, v) => data.map(attr => attr.name === k ? {...attr, value: v} : attr);

        return <sidebar.Tab className="modEditorDetailsTab">
            <sidebar.TabHeader>
                <gws.ui.Title content={this.props.editorFeature.label}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.sheet.Editor
                    data={this.props.editorData}
                    whenChanged={(name, value) => cc.update({
                        editorData: changed(name, value)
                    })}
                />
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.TextButton
                            primary
                            whenTouched={() => cc.saveDetails()}
                        >{this.__('modEditorSave')}</gws.ui.TextButton>
                    </Cell>
                </Row>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <Row>
                    <Cell>
                        <gws.ui.BackButton whenTouched={() => cc.hideDetails()}/>
                    </Cell>
                    <Cell flex/>
                </Row>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class SidebarBody extends gws.View<BodyProps> {
    render() {
        if (!this.props.editorLayer) {
            return <div className="modSidebarEmptyTab">
                {this.__('modEditorNoLayer')}
            </div>
        }

        return this.props.editorFeature
            ? <FeatureDetails {...this.props} />
            : <FeatureList {...this.props} />;

    }
}

class Controller extends gws.Controller implements gws.types.ISidebarItem {
    async init() {

        await this.app.addTool('EditorModifyTool', this.app.createController(ModifyTool, this));
        await this.app.addTool('EditorDrawTool', this.app.createController(DrawTool, this));
        await this.app.addTool('EditorPointerTool', this.app.createController(PointerTool, this));
    }

    get layer(): gws.types.IMapFeatureLayer {
        return this.app.store.getValue('editorLayer');
    }

    showDetails(f: gws.types.IMapFeature) {
        // let atts = f.props.attributes;
        // let data = this.layer.meta.attributes.map(a => ({
        //     name: a.name,
        //     value: atts[a.name] || '',
        //     type: a.type,
        //     editable: a.editable
        // } as gws.components.sheet.Attribute));
        //
        // this.highlight(f);
        // this.update({
        //     editorFeature: f,
        //     editorData: data
        // })
    }

    hideDetails() {
        this.update({
            editorFeature: null,
            editorData: null
        })
    }

    async saveDetails() {
        // let f = this.app.store.getValue('editorFeature') as gws.types.IMapFeature;
        // f.data = this.app.store.getValue('editorData').map(attr => ({
        //     [attr.name]: attr.value
        // }));
        //
        // let res = await this.app.server.editUpdateFeatures({
        //     layerUid: this.layer.uid,
        //     features: [f.params]
        // });
        // this.layer.reset();
    }

    highlight(f: gws.types.IMapFeature) {
        this.update({
            marker: {
                features: [f],
                mode: 'pan draw fade',
            }
        })
    }

    async addFeature(f: ol.Feature) {
        let feature = new gws.map.Feature(this.map, {geometry: f.getGeometry()});
        let res = await this.app.server.editAddFeatures({
            layerUid: this.layer.uid,
            features: [feature.props]
        });
        this.layer.reset();
    }

    async updateRaw(fs) {
        let features = fs.map(f => gws.map.Feature.fromOlFeature(this.map, f));
        console.log(features.map(f => f.uid))
        let res = await this.app.server.editUpdateFeatures({
            layerUid: this.layer.uid,
            features: features.map(f => f.params)
        });
        this.layer.reset();
    }

    selectFromEvent(evt: ol.MapBrowserPointerEvent) {
        let fs = this.layer.source.getFeaturesAtCoordinate(evt.coordinate);
        if (!fs)
            return;
        let feature = gws.map.Feature.fromOlFeature(this.map, fs[0])
        if (!feature)
            return;
        this.showDetails(feature);
    }

    tool = '';

    startTool(name) {
        this.app.startTool(this.tool = name);
    }

    stopTool() {
        this.app.stopTool(this.tool);
    }

    endEditing() {
        this.update({
            editorLayer: null,
            editorFeature: null,
            editorData: null,
        })
    }

    get iconClass() {
        return 'modEditorSidebarIcon';
    }

    get tooltip() {
        return this.__('modEditorTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarBody, [
                'mapUpdateCount',
                'editorLayer',
                'editorFeature',
                'editorData',

            ]),
            {map: this.map}
        );
    }

}

export const tags = {
    'Sidebar.Editor': Controller,
};
