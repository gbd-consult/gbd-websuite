import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';

let {Row, Cell} = gws.ui.Layout;

interface LayersViewProps extends gws.types.ViewProps {
    controller: LayersSidebar;
    mapUpdateCount: number;
    mapSelectedLayer?: gws.types.IMapLayer;
    layer?: gws.types.IMapLayer;
}

const LayersStoreKeys = [
    'mapUpdateCount',
    'mapSelectedLayer'
];

class LayersTreeTitle extends gws.View<LayersViewProps> {
    render() {
        let click = async () => {
            this.props.controller.map.deselectAllLayers();
            await this.props.controller.map.selectLayer(this.props.layer);
        };
        return <gws.ui.Button
            className="modLayersTreeTitle"
            whenTouched={click}
        >{this.props.layer.title}</gws.ui.Button>

    }
}

class LayersCheckButton extends gws.View<LayersViewProps> {
    render() {
        let layer = this.props.layer,
            isExclusive = layer.parent && layer.parent.exclusive,
            isChecked = layer.checked;

        let cls = gws.tools.cls(
            'modLayersCheckButton',
            layer.visible && 'isVisible',
            isExclusive && 'isExclusive',
            isChecked && 'isChecked'
        );

        return <gws.ui.IconButton
            {...cls}
            tooltip={this.__('modLayersCheckButton')}
            whenTouched={() => this.props.controller.map.setLayerChecked(layer, !(layer.visible && layer.checked))}
        />;
    }
}

class LayersExpandButton extends gws.View<LayersViewProps> {
    render() {
        let layer = this.props.layer,
            cls = layer.expanded ? 'modLayersCollapseButton' : 'modLayersExpandButton',
            fn = () => this.props.controller.map.setLayerExpanded(layer, !layer.expanded);

        return <gws.ui.IconButton
            className={cls}
            tooltip={this.__('modLayersExpandButton')}
            whenTouched={fn}
        />
    }
}

class LayersLeafButton extends gws.View<LayersViewProps> {
    render() {
        return <gws.ui.IconButton
            className='modLayersLeafButton'
            tooltip={this.__('modLayersLeafButton')}
        />;
    }
}

let _layerTree = (layer: gws.types.IMapLayer, props) => {
    let cc = [];

    layer.children.forEach(la => {
        if (!la.shouldList)
            return;
        if (la.unfolded)
            cc.push(..._layerTree(la, props));
        else
            cc.push(<LayersTreeNode key={la.uid} {...props} layer={la}/>)
    });

    return cc.length ? cc : null;
};

class LayersTreeNode extends gws.View<LayersViewProps> {
    render() {

        let layer = this.props.layer,
            children = _layerTree(layer, this.props);

        return <div className="modLayersTreeNode">
            <Row {...gws.tools.cls('modLayersTreeRow', layer.visible && 'isVisible', layer.selected && 'isSelected')}>
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

class LayerSidebarDetails extends gws.View<LayersViewProps> {
    render() {
        let layer = this.props.layer,
            cc = this.props.controller,
            map = cc.map;

        let f = {
            zoom() {
                map.setViewExtent(layer.extent, true)
            },
            show() {
                map.hideAllLayers();
                map.setLayerChecked(layer, true)
            },
            edit() {
                cc.update({
                    editLayer: layer,
                    editFeature: null,
                    sidebarActiveTab: 'Sidebar.Edit',
                });
            },
            close() {
                map.deselectAllLayers()
            }
        };

        return <div className="modLayersDetails">
            <div className="modLayersDetailsBody">
                <div className="modLayersDetailsBodyContent">
                    <gws.components.Description content={this.props.layer.description}/>
                </div>
            </div>

            <sidebar.AuxToolbar>
                <Cell flex/>
                <sidebar.AuxButton
                    className="modLayersZoomAuxButton"
                    tooltip={this.__('modLayersZoomAuxButton')}
                    whenTouched={f.zoom}
                />
                <sidebar.AuxButton
                    className="modLayersShowAuxButton"
                    tooltip={this.__('modLayersShowAuxButton')}
                    whenTouched={f.show}
                />
                {layer.editAccess && <sidebar.AuxButton
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

class LayersSidebarView extends gws.View<LayersViewProps> {
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
            this.connect(LayersSidebarView, LayersStoreKeys),
            {map: this.map}
        );
    }
}

export const tags = {
    'Sidebar.Layers': LayersSidebar
};

