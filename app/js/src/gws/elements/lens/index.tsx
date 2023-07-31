import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as components from 'gws/components';

import * as draw from 'gws/elements/draw';
import * as toolbar from 'gws/elements/toolbar';
import * as toolbox from 'gws/elements/toolbox';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Lens';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as Controller;

interface ViewProps extends gws.types.ViewProps {
    controller: Tool;
    lensShapeType: string;
}

const StoreKeys = [
    'lensShapeType',
];

class LensLayer extends gws.map.layer.FeatureLayer {
    get printPlane() {
        return null;
    }
}

class ToolboxView extends gws.View<ViewProps> {
    render() {
        let button = (type, cls, tooltip) => {
            return <Cell>
                <gws.ui.Button
                    {...gws.lib.cls(cls, type === this.props.lensShapeType && 'isActive')}
                    tooltip={tooltip}
                    whenTouched={() => this.props.controller.setShapeType(type)}
                />
            </Cell>
        };

        let buttons = [
            button('Point', 'modDrawPointButton', this.__('modDrawPointButton')),
            button('Line', 'modDrawLineButton', this.__('modDrawLineButton')),
            button('Box', 'modDrawBoxButton', this.__('modDrawBoxButton')),
            button('Polygon', 'modDrawPolygonButton', this.__('modDrawPolygonButton')),
            button('Circle', 'modDrawCircleButton', this.__('modDrawCircleButton')),
        ];

        return <toolbox.Content
            controller={this.props.controller}
            buttons={buttons}
        />

    }

}

export class Tool extends gws.Tool {

    layerPtr: LensLayer;
    oOverlay: ol.Overlay;
    overlayRef: React.RefObject<HTMLDivElement>;
    ixDraw: ol.interaction.Draw;
    ixModify: ol.interaction.Modify;
    drawState: string = '';
    styleName = '.lensFeature';

    async init() {
        this.overlayRef = React.createRef();

        this.app.whenCalled('lensStartFromFeature', args => {
            this.app.startTool('Tool.Lens');
            this.reset();
            this.createFeature(args.feature.geometry.clone());
            this.createOverlay();
            this.run();
        });

        this.update({lensShapeType: 'Box'});
    }

    get toolboxView() {
        return this.createElement(
            this.connect(ToolboxView, StoreKeys)
        );
    }

    start() {
        this.reset();

        let drawFeature;

        this.ixDraw = this.map.drawInteraction({
            shapeType: this.getValue('lensShapeType'),
            style: this.styleName,
            whenStarted: (oFeatures) => {
                drawFeature = oFeatures[0];
                this.drawState = 'drawing';
            },
            whenEnded: () => {
                if (this.drawState === 'drawing') {
                    this.createFeature(drawFeature.getGeometry());
                    this.createOverlay();
                    this.run();
                }
                this.drawState = '';
            },
        });

        this.map.appendInteractions([this.ixDraw]);
    }

    stop() {
        this.reset();
    }

    //

    get layer() {
        if (!this.layerPtr) {
            this.layerPtr = this.map.addServiceLayer(new LensLayer(this.map, {
                uid: '_lens',
                cssSelector: this.styleName,
            }));
        }
        return this.layerPtr;

    }

    get feature(): gws.types.IFeature {
        if (!this.layerPtr)
            return null;
        let fs = this.layer.features;
        return fs.length ? fs[0] : null;
    }

    async run() {
        if (!this.feature)
            return;
        await this.runSearch(this.feature.geometry);
    }

    async runSearch(geometry) {
        let features = await this.map.searchForFeatures({geometry});

        if (features.length) {
            this.update({
                marker: {
                    features,
                    mode: 'draw',
                },
                infoboxContent: <components.feature.InfoList controller={this} features={features}/>
            });
        } else {
            this.update({
                marker: {
                    features: null,
                },
                infoboxContent: null
            });
        }
    }

    setShapeType(st) {
        this.drawCancel();
        this.update({lensShapeType: st});
        this.start();
    }

    drawCancel() {
        if (this.drawState === 'drawing') {
            this.drawState = 'cancel';
            this.ixDraw.finishDrawing();
        }
    }

    reset() {
        if (this.layerPtr)
            this.map.removeLayer(this.layerPtr);
        this.layerPtr = null;
        this.removeOverlay();
        this.map.resetInteractions();
        this.ixModify = this.ixDraw = null;
    }

    createFeature(geom: ol.geom.Geometry) {
        this.layer.clear();
        let f = this.app.models.defaultModel().featureFromGeometry(geom);
        this.layer.addFeature(f);
        this.layer.show();
        this.ixModify = this.map.modifyInteraction({
            layer: this.layer,
            whenEnded: () => this.run()
        });
        this.map.appendInteractions([this.ixDraw, this.ixModify]);



    }

    createOverlay() {
        this.removeOverlay();

        // https://github.com/openlayers/openlayers/issues/6948
        // there's a problem in OL with react events in overlays
        // so let's do it the old way

        let div = document.createElement('div');
        div.className = 'lensOverlay';

        let button = document.createElement('div');
        button.className = 'uiIconButton lensOverlayAnchorButton';
        button.title = this.__('lensOverlayAnchorButton');
        button.addEventListener('mousedown', evt => this.overlayMoveTouched(evt))

        div.appendChild(button);

        this.oOverlay = new ol.Overlay({
            element: div,
            stopEvent: true,
            positioning: 'center-center',
        });

        this.map.oMap.addOverlay(this.oOverlay);
        this.positionOverlay();
    }

    removeOverlay() {
        if (this.oOverlay)
            this.map.oMap.removeOverlay(this.oOverlay);
        this.oOverlay = null;
    }

    overlayMoveTouched(evt) {
        if (!this.feature)
            return;

        gws.lib.trackDrag({
            map: this.map,
            whenMoved: px => {
                let a = this.oOverlay.getPosition(),
                    b = this.map.oMap.getCoordinateFromPixel(px);

                let geom = this.feature.geometry as ol.geom.SimpleGeometry;

                geom.translate(
                    b[0] - a[0],
                    b[1] - a[1],
                );

                this.positionOverlay();

            },
            whenEnded: () => this.run()
        })
    }

    positionOverlay() {
        if (!this.feature)
            return;
        let ext = this.feature.geometry.getExtent();
        let pos = ol.extent.getCenter(ext);
        this.oOverlay.setPosition(pos);
    }

}

class ToolbarButton extends toolbar.Button {
    iconClass = 'lensToolbarButton';
    tool = 'Tool.Lens';

    get tooltip() {
        return this.__('modToolbarButton');
    }
}

class Controller extends gws.Controller implements gws.types.IController {
    uid = MASTER;

    toolTag: string;

    get tool(): Tool {
        return this.app.tool(this.toolTag) as Tool;
    }

    oFeature: ol.Feature;

    start(tag: string) {
        this.toolTag = tag;

    }

    stop(tool) {

    }

    drawEnded() {
        //this.drawTool.stop();
        this.app.stopTool('Tool.Lens.Draw');
        this.app.startTool(this.toolTag);
    }

}

gws.registerTags({
    [MASTER]: Controller,
    'Toolbar.Lens': ToolbarButton,
    'Tool.Lens': Tool,
});
