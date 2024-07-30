import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as types from './types';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;


export class GeometryTextDialog extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    save() {
        let cc = this.master();
        let dd = cc.editState.dialogData as types.GeometryTextDialogData;
        dd.whenSaved(dd.shape);
    }

    toWKT(shape: gws.api.base.shape.Props): string {
        let cc = this.master();
        const wktFormat = new ol.format.WKT();
        let geom = cc.map.shape2geom(shape);
        return wktFormat.writeGeometry(geom);
    }

    updateShape(shape: gws.api.base.shape.Props) {
        let cc = this.master();
        let dd = cc.editState.dialogData as types.GeometryTextDialogData;

        cc.updateEditState({
            dialogData: {...dd, shape}
        });
    }

    updateFromWKT(wkt: string) {
        let cc = this.master();
        const wktFormat = new ol.format.WKT();
        let geom = wktFormat.readGeometry(wkt);
        let shape = cc.map.geom2shape(geom);
        this.updateShape(shape);
    }

    updatePointCoordinate(index: number, value: string) {
        let n = parseFloat(value);
        if (Number.isNaN(n)) {
            return
        }

        let cc = this.master();
        let dd = cc.editState.dialogData as types.GeometryTextDialogData;
        let shape = {
            crs: dd.shape.crs,
            geometry: {
                type: dd.shape.geometry.type,
                coordinates: [...dd.shape.geometry.coordinates]
            }
        };
        shape.geometry.coordinates[index] = n;
        this.updateShape(shape);
    }


    render() {
        let cc = this.master();
        let dd = cc.editState.dialogData as types.GeometryTextDialogData;

        let okButton = <gws.ui.Button
            {...gws.lib.cls('editSaveButton', 'isActive')}
            tooltip={this.__('editSave')}
            whenTouched={() => this.save()}
        />


        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;


        return <gws.ui.Dialog
            className="editGeometryTextDialog"
            title={this.__('editGeometryTextTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[okButton, cancelButton]}
        >
            <Form>
                {dd.shape && dd.shape.geometry.type === 'Point' && <Row>
                    <Cell>
                        <gws.ui.TextInput
                            label="X"
                            value={cc.map.formatCoordinate(dd.shape.geometry.coordinates[0])}
                            whenChanged={v => this.updatePointCoordinate(0, v)}
                            whenEntered={() => this.save()}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.TextInput
                            label="Y"
                            value={cc.map.formatCoordinate(dd.shape.geometry.coordinates[1])}
                            whenChanged={v => this.updatePointCoordinate(1, v)}
                            whenEntered={() => this.save()}
                        />
                    </Cell>
                </Row>}
                <Row>
                    <Cell flex>
                        <gws.ui.TextArea
                            label="WKT"
                            value={this.toWKT(dd.shape)}
                            whenChanged={v => this.updateFromWKT(v)}
                            height={200}
                        />
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}
