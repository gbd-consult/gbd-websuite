import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as components from 'gws/components';

let {Form, Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: LayersSidebar;
    mapUpdateCount: number;
    mapSelectedLayer?: gws.types.ILayer;
    modLayersOpacityVisible: boolean;
    layer?: gws.types.ILayer;
}

const StoreKeys = [
    'mapUpdateCount',
    'mapSelectedLayer',
    'modLayersOpacityVisible',
    'layersShowInactive',
];

class LayersTreeTitle extends gws.View<ViewProps> {
    render() {
        let click = async () => {
            this.props.controller.map.deselectAllLayers();
            await this.props.controller.map.selectLayer(this.props.layer);
        };

        return <gws.ui.Touchable
            className="modLayersTreeTitle"
            whenTouched={click}
        >{this.props.layer.title}</gws.ui.Touchable>

    }
}

class LayersCheckButton extends gws.View<ViewProps> {
    render() {
        let layer = this.props.layer,
            isExclusive = layer.parent && layer.parent.exclusive,
            isChecked = layer.checked,
            isGroup = !gws.lib.isEmpty(layer.children);

        if (!layer.inResolution) {
            return <gws.ui.Button
                className='modLayersCheckButton isInactive'
                tooltip={this.__('modLayersCheckButton')}
            />;
        }

        let cls = gws.lib.cls(
            'modLayersCheckButton',
            layer.hidden && 'isHidden',
            isExclusive && 'isExclusive',
            isChecked && 'isChecked',
            isGroup && 'isGroup',
        );

        return <gws.ui.Button
            {...cls}
            tooltip={this.__('modLayersCheckButton')}
            whenTouched={() => this.props.controller.map.setLayerChecked(layer, !(!layer.hidden && layer.checked))}
        />;
    }
}

class LayersExpandButton extends gws.View<ViewProps> {
    render() {
        let layer = this.props.layer,
            cls = layer.expanded ? 'modLayersCollapseButton' : 'modLayersExpandButton',
            fn = () => this.props.controller.map.setLayerExpanded(layer, !layer.expanded);

        return <gws.ui.Button
            className={cls}
            tooltip={this.__('modLayersExpandButton')}
            whenTouched={fn}
        />
    }
}

class LayersLeafButton extends gws.View<ViewProps> {
    render() {
        return <gws.ui.Button
            className='modLayersLeafButton'
            tooltip={this.__('modLayersLeafButton')}
        />;
    }
}

let _layerTree = (layer: gws.types.ILayer, props) => {
    let cc = [];

    layer.children.forEach(la => {
        if (la.unlisted)
            return;
        if (!la.inResolution && !props.layersShowInactive)
            return;
        if (la.unfolded)
            cc.push(..._layerTree(la, props));
        else
            cc.push(<LayersTreeNode key={la.uid} {...props} layer={la}/>)
    });

    return cc.length ? cc : null;
};

class LayersTreeNode extends gws.View<ViewProps> {
    render() {

        let layer = this.props.layer,
            children = _layerTree(layer, this.props),
            cls = gws.lib.cls(
                'modLayersTreeRow',
                (layer.hidden || !layer.inResolution) && 'isHidden',
                layer.selected && 'isSelected',
                !layer.inResolution && 'isInactive',
            );

        return <div className="modLayersTreeNode">
            <Row {...cls}>
                <Cell>
                    {children
                        ? <LayersExpandButton {...this.props}  />
                        : <LayersLeafButton {...this.props}  />
                    }
                </Cell>
                <Cell flex>
                    <LayersTreeTitle {...this.props} />
                </Cell>
                <Cell>
                    <LayersCheckButton {...this.props} />
                </Cell>
            </Row>
            {children && layer.expanded && <div className="modLayersTreeChildren">{children}</div>}
        </div>
    }
}

const ZOOM_EXTENT_PADDING = 50;

class LayerSidebarDetails extends gws.View<ViewProps> {
    render() {
        let layer = this.props.layer,
            cc = this.props.controller,
            map = cc.map;

        // let models = this.app.modelRegistry.editableModels().filter(m => m.layer === layer);
        // let model = models.length ? models[0] : null;

        let model = null;


        let f = {
            zoom() {
                console.log('ZOOM_TO_LAYER', layer.uid, layer.extent);
                map.setViewExtent(layer.extent, true, ZOOM_EXTENT_PADDING);
            },
            show() {
                map.hideAllLayers();
                map.setLayerChecked(layer, true)
            },
            edit() {
                cc.app.call('editModel', {model})
            },
            close() {
                map.deselectAllLayers()
            },
            setOpacity(v) {
                layer.opacity = v / 100;
                map.computeOpacities();
            },
            toggleOpacityControl() {
                cc.update({
                    modLayersOpacityVisible: !cc.getValue('modLayersOpacityVisible')
                })
            }
        };

        return <div className="modLayersDetails">
            <div className="modLayersDetailsBody">
                <div className="modLayersDetailsBodyContent">
                    <components.Description content={this.props.layer.description}/>
                </div>
            </div>
            {this.props.modLayersOpacityVisible && <div className="modLayersDetailsControls">
                <Form>
                    <Row>
                        <Cell>
                            {this.__('modLayersOpacity')}
                        </Cell>
                        <Cell flex>
                            <gws.ui.Slider
                                value={Math.floor(100 * this.props.layer.opacity)}
                                minValue={0}
                                maxValue={100}
                                step={1}
                                whenChanged={f.setOpacity}
                            />
                        </Cell>
                        <Cell width={50}>
                            <gws.ui.NumberInput
                                value={Math.floor(100 * this.props.layer.opacity)}
                                minValue={0}
                                maxValue={100}
                                whenChanged={f.setOpacity}
                            />
                        </Cell>
                    </Row>
                </Form>
            </div>}

            <sidebar.AuxToolbar>
                <Cell flex/>
                <sidebar.AuxButton
                    className="modLayersOpacityAuxButton"
                    tooltip={this.__('modLayersOpacityAuxButton')}
                    whenTouched={f.toggleOpacityControl}
                />
                <sidebar.AuxButton
                    className="modLayersZoomAuxButton"
                    tooltip={this.__('modLayersZoomAuxButton')}
                    whenTouched={f.zoom}
                />
                {model && <sidebar.AuxButton
                    className="modLayersEditAuxButton"
                    tooltip={this.__('modLayersEditAuxButton')}
                    whenTouched={f.edit}
                />}
                <sidebar.AuxCloseButton
                    tooltip={this.__('modLayersCloseAuxButton')}
                    whenTouched={f.close}
                />
            </sidebar.AuxToolbar>
        </div>
    }
}

class LayersSidebarView extends gws.View<ViewProps> {
    render() {
        let sel = this.props.mapSelectedLayer;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modLayersSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {_layerTree(this.props.controller.map.root, this.props)}
            </sidebar.TabBody>

            {sel && <sidebar.TabFooter>
                <LayerSidebarDetails {...this.props} layer={sel}/>
            </sidebar.TabFooter>}
        </sidebar.Tab>
    }
}

class LayersSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modLayersSidebar';

    get tooltip() {
        return this.__('modLayersSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(LayersSidebarView, StoreKeys),
            {map: this.map}
        );
    }
}

gws.registerTags({
    'Sidebar.Layers': LayersSidebar
});

