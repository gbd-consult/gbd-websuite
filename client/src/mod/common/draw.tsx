import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Draw';

export interface DrawCallback {
    whenStarted: (shapeType: string, oFeature: ol.Feature) => void;
    whenEnded: (shapeType: string, oFeature: ol.Feature) => void;
    whenCancelled: () => void;
}

interface DrawProps extends gws.types.ViewProps {
    controller: DrawController;
    drawShapeType: string;
    drawCallback: DrawCallback;
    drawStyle?: gws.types.IMapStyle;
    drawMode: boolean;
}

const DrawPropsKeys = [
    'drawShapeType',
    'drawCallback',
    'drawStyle',
    'drawMode',

];

const DEFAULT_SHAPE_TYPE = 'Polygon';


export class Tool extends gws.Controller implements gws.types.ITool {
    style: gws.types.IMapStyle;

    start() {
        let master = this.app.controller(MASTER) as DrawController;
        master.start(this);
    }

    stop() {
        let master = this.app.controller(MASTER) as DrawController;
        master.stop();
    }

    whenStarted(shapeType, oFeature) {}
    whenEnded(shapeType, oFeature) {}
    whenCancelled() {}


}

class DrawBox extends gws.View<DrawProps> {

    render() {
        let cc = this.props.controller;

        console.log('DrawBox', this.props.drawMode)

        if (!this.props.drawMode)
            return <div className="modDrawControlBox"/>;

        let shapeType = this.props.drawShapeType || DEFAULT_SHAPE_TYPE

        let button = (type, cls, tooltip) => {
            return <Cell>
                <gws.ui.IconButton
                    {...gws.tools.cls(cls, type === shapeType && 'isActive')}
                    tooltip={tooltip}
                    whenTouched={() => cc.setShapeType(type)}
                />
            </Cell>
        };

        return <div className="modDrawControlBox isActive">
            <Row>
                {button('Point', 'modDrawPointButton', cc.__('modDrawPointButton'))}
                {button('Line', 'modDrawLineButton', cc.__('modDrawLineButton'))}
                {button('Box', 'modDrawBoxButton', cc.__('modDrawBoxButton'))}
                {button('Polygon', 'modDrawPolygonButton', cc.__('modDrawPolygonButton'))}
                {button('Circle', 'modDrawCircleButton', cc.__('modDrawCircleButton'))}
                <Cell flex/>
                <Cell>
                    <gws.ui.IconButton
                        className="cmpButtonFormOk"
                        tooltip={this.props.controller.__('modDrawOkButton')}
                        whenTouched={() => this.props.controller.commit()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        className="cmpButtonFormCancel"
                        tooltip={this.props.controller.__('modDrawCancelButton')}
                        whenTouched={() => this.props.controller.cancel()}
                    />
                </Cell>
            </Row>
        </div>;
    }
}


class DrawController extends gws.Controller {
    uid = MASTER;
    oInteraction: ol.interaction.Draw;
    oFeature: ol.Feature;
    state: string;
    currTool: Tool;

    get appOverlayView() {
        return this.createElement(
            this.connect(DrawBox, DrawPropsKeys));
    }

    get shapeType() {
        return this.getValue('drawShapeType') || DEFAULT_SHAPE_TYPE;
    }


    start(tool) {
        this.currTool = tool;
        this.startDrawing();
    }

    startDrawing() {

        let shapeType = this.shapeType;

        console.log('START_DRAW', this.shapeType)

        this.oFeature = null;
        this.state = 'started';

        this.oInteraction = this.map.drawInteraction({
            shapeType,
            style: this.currTool.style,
            whenStarted: (oFeatures) => {
                console.log('DRAW whenStarted', this.state);
                this.oFeature = oFeatures[0];
                this.state = 'drawing';
                this.currTool.whenStarted(this.shapeType, this.oFeature);
            },
            whenEnded: () => {
                console.log('DRAW whenEnded', this.state);
                if (this.state === 'drawing') {
                    this.state = 'ended';
                    this.stopDrawing();
                }
            },
        });

        this.map.setExtraInteractions([this.oInteraction]);

        this.update({
            drawMode: true,
            drawShapeType: shapeType
        });

    }

    stop() {
        this.stopDrawing();
        this.update({
            drawMode: false
        })

    }

    stopDrawing() {
        console.log('stopDrawing', this.state);

        let state = this.state;

        this.state = '';

        if (state === 'ended') {
            this.currTool.whenEnded(this.shapeType, this.oFeature);
        }
    }

    commit() {
        if (this.state === 'drawing') {
            this.oInteraction.finishDrawing();
        } else {
            this.stopDrawing();
        }
    }

    cancel() {
        this.stopDrawing();
        this.currTool.whenCancelled();

    }

    setShapeType(s) {
        this.update({
            drawShapeType: s
        });
        this.startDrawing();
    }

    turnOff() {
        this.update({
            drawMode: false
        });

    }

    invokeCallback(key, ...args) {
        let cb = this.getValue('drawCallback');

        if (cb && cb[key])
            cb[key](...args);

    }

}

export const tags = {
    [MASTER]: DrawController,
};
