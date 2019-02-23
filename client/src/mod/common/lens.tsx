import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as draw from '../common/draw';
import * as toolbar from '../common/toolbar';
import * as toolbox from '../common/toolbox';

const MASTER = 'Shared.Lens';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as LensController;

class LensLayer extends gws.map.layer.FeatureLayer {
    get printItem() {
        return null;
    }

}

export class LensTool extends gws.Tool {

    get _toolboxView() {
        return <toolbox.Content
            controller={this}
            iconClass="modIdentifyClickToolboxIcon"
            title={this.__('modIdentifyClickToolboxTitle')}
            hint={this.__('modIdentifyClickToolboxHint')}
            buttons={[
                <gws.ui.IconButton
                    className="modToolboxCancelButton"
                    tooltip={this.__('modDrawCancelButton')}
                    whenTouched={() => _master(this).overlayDrawTouched(null)}
                />,

            ]}
        />
    }

    get style() {
        return this.map.getStyleFromSelector('.modLensFeature');
    }

    get editStyle() {
        return this.map.getStyleFromSelector('.modLensFeatureEdit');
    }

    start() {
        _master(this).start(this.tag);
    }

    stop() {
        _master(this).stop(this);
    }

    async whenChanged(geometry) {
        let features = await this.map.searchForFeatures({geometry});

        if (features.length) {
            this.update({
                marker: {
                    features: [features[0]],
                    mode: 'draw',
                },
                infoboxContent: <gws.components.feature.InfoList controller={this} features={features}/>
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

}

class LensDrawTool extends draw.Tool {
    whenStarted(shapeType, oFeature) {
    }

    whenEnded(shapeType, oFeature) {
        _master(this).update({
            lensGeometry: oFeature.getGeometry()
        })
        _master(this).drawEnded();
    }

    whenCancelled() {
        _master(this).drawEnded();
    }
}

class LensToolbarButton extends toolbar.Button {
    iconClass = 'modLensToolbarButton';
    tool = 'Tool.Lens';

    get tooltip() {
        return this.__('modLensToolbarButton');
    }
}

class LensController extends gws.Controller implements gws.types.IController {
    uid = MASTER;

    toolTag: string;
    layer: LensLayer;
    oOverlay: ol.Overlay;
    overlayRef: React.RefObject<HTMLDivElement>;

    get feature(): gws.types.IMapFeature {
        if (!this.layer)
            return null;
        let fs = this.layer.features;
        return fs.length ? fs[0] : null;
    }

    get tool(): LensTool {
        return this.app.tool(this.toolTag) as LensTool;
    }

    async init() {
        this.overlayRef = React.createRef();

        this.app.whenCalled('lensStartFromFeature', args => {
            this.update({
                lensGeometry: args.feature.geometry.clone(),
            });
            this.app.startTool('Tool.Lens');
        });

    }

    DEFAULT_GEOM_SIZE = 100;

    get defaultGeometry() {
        let vs = this.map.viewState;
        let px = this.map.oMap.getPixelFromCoordinate([vs.centerX, vs.centerY]);
        let cc = this.map.oMap.getCoordinateFromPixel([px[0] + this.DEFAULT_GEOM_SIZE, px[1]]);

        return new ol.geom.Circle(
            [vs.centerX, vs.centerY], cc[0] - vs.centerX);

    }

    start(tag: string) {
        this.toolTag = tag;

        this.layer = this.map.addServiceLayer(new LensLayer(this.map, {
            uid: '_lens',
            style: this.tool.style,
        }));

        let geom = this.getValue('lensGeometry') || this.defaultGeometry;

        let extent = geom.getExtent();
        this.map.setCenter(ol.extent.getCenter(extent), true);

        this.createFeature(geom);
        this.createOverlay();
        this.layer.show();

        let ixModify = this.map.modifyInteraction({
            layer: this.layer,
            whenEnded: () => this.run()
        });

        this.map.setExtraInteractions([ixModify]);

        this.run()

    }

    stop(tool) {
        this.map.removeLayer(this.layer);
        this.removeOverlay();
        this.map.setExtraInteractions([]);

    }

    run() {
        if (!this.feature) {
            console.log('LENS_RUN', 'no feature');
            return;
        }

        let geom = this.feature.geometry;

        this.update({
            lensGeometry: geom
        })

        console.log('LENS_RUN', this.feature.geometry)

        this.tool.whenChanged(geom);
    }

    createOverlay() {
        // https://github.com/openlayers/openlayers/issues/6948
        // there's a problem in OL with react events in overlays
        // so let's do it the old way

        let div = document.createElement('div');

        let buttons = [
            document.createElement('div'),
            document.createElement('div'),
            document.createElement('div')
        ];

        div.className = 'modLensOverlay';

        buttons[0].className = 'uiIconButton modLensOverlayDrawButton';
        buttons[1].className = 'uiIconButton modLensOverlayAnchorButton';
        buttons[2].className = 'uiIconButton modLensOverlayCancelButton';

        buttons[0].title = this.__('modLensOverlayDrawButton');
        buttons[1].title = this.__('modLensOverlayAnchorButton');
        buttons[2].title = this.__('modLensOverlayCancelButton');

        buttons[0].addEventListener('click', evt => this.overlayDrawTouched(evt))
        buttons[1].addEventListener('mousedown', evt => this.overlayMoveTouched(evt))
        buttons[2].addEventListener('click', evt => this.overlayCloseTouched(evt))

        buttons.forEach(b => div.appendChild(b))

        this.oOverlay = new ol.Overlay({
            element: div,
            stopEvent: true,
            positioning: 'center-center',
        });

        this.map.oMap.addOverlay(this.oOverlay);
        this.positionOverlay();
    }

    createFeature(geom: ol.geom.Geometry) {
        this.layer.clear();
        let f = new gws.map.Feature(this.map, {geometry: geom});
        this.layer.addFeature(f);
        this.run();
    }

    overlayCloseTouched(evt) {
        this.app.startTool('Tool.Default');
    }

    overlayDrawTouched(evt) {
        let drawTool = this.app.tool('Tool.Lens.Draw') as LensDrawTool;
        drawTool.style = this.tool.editStyle;
        this.app.startTool('Tool.Lens.Draw');
    }

    removeOverlay() {
        if (this.oOverlay)
            this.map.oMap.removeOverlay(this.oOverlay);
        this.oOverlay = null;
    }

    drawEnded() {
        //this.drawTool.stop();
        this.app.stopTool('Tool.Lens.Draw');
        this.app.startTool(this.toolTag);
    }

    clear() {
        this.layer.clear();
        if (this.oOverlay)
            this.map.oMap.removeOverlay(this.oOverlay);
    }

    overlayMoveTouched(evt) {
        if (!this.feature)
            return;

        gws.tools.trackDrag({
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

export const tags = {
    [MASTER]: LensController,
    'Toolbar.Lens': LensToolbarButton,
    'Tool.Lens': LensTool,
    'Tool.Lens.Draw': LensDrawTool,
};
