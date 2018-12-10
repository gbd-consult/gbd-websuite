import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from './sidebar';

let {Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: SidebarLayersController;
    mapUpdateCount: number;
    mapSelectedLayer?: gws.types.IMapLayer;
    layer?: gws.types.IMapLayer;
}

class Title extends gws.View<ViewProps> {
    render() {
        let click = async () => {
            this.props.controller.map.deselectAllLayers();
            await this.props.controller.map.selectLayer(this.props.layer);
        };
        return <gws.ui.ItemButton
            className="modLayersLayerTitle"
            whenTouched={click}
        >{this.props.layer.title}</gws.ui.ItemButton>

    }
}

class ToggleVisibleButton extends gws.View<ViewProps> {
    render() {
        let layer = this.props.layer,
            cls = layer.visible ? 'modLayersHideButton' : 'modLayersShowButton',
            fn = () => this.props.controller.map.setLayerVisible(layer, !layer.visible);

        return <gws.ui.IconButton
            className={cls}
            tooltip={this.__('modLayersVisibleButton')}
            whenTouched={fn}
        />;
    }
}

class ToggleExpandButton extends gws.View<ViewProps> {
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

let _childNodes = (layer: gws.types.IMapLayer, props) => {
    let cc = [];

    layer.children.forEach(la => {
        if (!la.shouldList)
            return;
        if (la.unfolded)
            cc.push(..._childNodes(la, props));
        else
            cc.push(<TreeNode key={la.uid} {...props} layer={la}/>)
    });

    return cc.length ? cc : null;
};

class TreeNode extends gws.View<ViewProps> {
    render() {

        let layer = this.props.layer,
            children = _childNodes(layer, this.props);

        return <div className="modLayersContainer">
            <Row {...gws.tools.cls('modLayersLayer', layer.visible && 'visible', layer.selected && 'isSelected')}>
                <Cell>
                    {children
                        ? <ToggleExpandButton {...this.props}  />
                        : <gws.ui.IconButton
                            className="modLayersLayerButton"
                            tooltip={this.__('modLayersLayerButton')}
                        />
                    }
                </Cell>
                <Cell flex>
                    <Title {...this.props} />
                </Cell>
                <Cell>
                    <ToggleVisibleButton {...this.props} />
                </Cell>
            </Row>
            {children && layer.expanded && <div className="modLayersChildren">{children}</div>}
        </div>
    }
}

class LayerDetailsToolbar extends gws.View<ViewProps> {
    render() {
        let layer = this.props.layer,
            ctrl = this.props.controller,
            map = ctrl.map;

        let f = {
            zoom() {
                map.setViewExtent(layer.extent, true)
            },
            show() {
                map.hideAllLayers();
                map.setLayerVisible(layer, true)
            },
            edit() {
                ctrl.update({
                    editorLayer: layer,
                    sidebarActiveTab: 'Sidebar.Editor',
                });
            },
            close() {
                map.deselectAllLayers()
            }
        };

        return <Row>
            <Cell flex/>
            <Cell>
                <gws.ui.IconButton
                    className="modLayersDetailsZoomButton"
                    tooltip={this.__('modLayersDetailsZoomButton')}
                    whenTouched={f.zoom}
                />
            </Cell>
            <Cell>
                <gws.ui.IconButton
                    className="modLayersDetailsShowButton"
                    tooltip={this.__('modLayersDetailsShowButton')}
                    whenTouched={f.show}
                />
            </Cell>
            {layer.editable && <Cell>
                <gws.ui.IconButton
                    className="modLayersDetailsEditButton"
                    tooltip={this.__('modLayersDetailsEditButton')}
                    whenTouched={f.edit}
                />
            </Cell>}
            <Cell>
                <gws.ui.IconButton
                    className="modLayersDetailsCloseButton"
                    tooltip={this.__('modLayersDetailsCloseButton')}
                    whenTouched={f.close}
                />
            </Cell>
        </Row>;

    }
}

class LayerDetails extends gws.View<ViewProps> {
    render() {
        return <div className="modLayersDetails">
            <div className="modLayersDetailsBody">
                <div className="modLayersDetailsBodyContent">
                    <gws.ui.TextBlock className="cmpDescription" withHTML content={this.props.layer.description}/>
                </div>
            </div>
            <div className="modSidebarSecondaryToolbar">
                <LayerDetailsToolbar {...this.props} />
            </div>
        </div>
    }
}

class SidebarBody extends gws.View<ViewProps> {
    render() {
        let selectedLayer = this.props['mapSelectedLayer'];

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modLayersTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {_childNodes(this.props.controller.map.root, this.props)}
            </sidebar.TabBody>

            {selectedLayer && <sidebar.TabFooter>
                <LayerDetails {...this.props} layer={selectedLayer}/>
            </sidebar.TabFooter>}
        </sidebar.Tab>
    }
}

class SidebarLayersController extends gws.Controller implements gws.types.ISidebarItem {

    selectLayer(layer) {
        this.update({
            selectedLayer: layer
        })
    }

    editSelected() {
        let la = this.app.store.getValue('mapSelectedLayer');
        if (la && la.config.editable) {
            this.update({
                editorLayer: la,
                sidebarActiveTab: 'Sidebar.Editor',
            })
        }
    }

    get iconClass() {
        return 'modLayersSidebarIcon'
    }

    get tooltip() {
        return this.__('modLayersTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarBody, ['mapUpdateCount', 'mapSelectedLayer']),
            {map: this.map}
        );
    }

}

export const tags = {
    'Sidebar.Layers': SidebarLayersController
};

