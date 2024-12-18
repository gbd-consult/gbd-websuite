import * as React from 'react';
import * as ol from 'openlayers';

import * as gc from 'gc';
;
import * as types from './types';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gc.ui.Layout;


export class GeometryTextDialog extends gc.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    save() {
        let cc = this.master();
        let dd = cc.editState.dialogData as types.GeometryTextDialogData;
        dd.whenSaved(dd.shape);
    }

    toWKT(shape: gc.gws.base.shape.Props): string {
        let cc = this.master();
        const wktFormat = new ol.format.WKT();
        let geom = cc.map.shape2geom(shape);
        return wktFormat.writeGeometry(geom);
    }

    updateShape(shape: gc.gws.base.shape.Props) {
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

        let okButton = <gc.ui.Button
            {...gc.lib.cls('editSaveButton', 'isActive')}
            tooltip={this.__('editSave')}
            whenTouched={() => this.save()}
        />


        let cancelButton = <gc.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;


        return <gc.ui.Dialog
            className="editGeometryTextDialog"
            title={this.__('editGeometryTextTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[okButton, cancelButton]}
        >
            <Form>
                {dd.shape && dd.shape.geometry.type === 'Point' && <Row>
                    <Cell>
                        <gc.ui.TextInput
                            label="X"
                            value={cc.map.formatCoordinate(dd.shape.geometry.coordinates[0])}
                            whenChanged={v => this.updatePointCoordinate(0, v)}
                            whenEntered={() => this.save()}
                        />
                    </Cell>
                    <Cell>
                        <gc.ui.TextInput
                            label="Y"
                            value={cc.map.formatCoordinate(dd.shape.geometry.coordinates[1])}
                            whenChanged={v => this.updatePointCoordinate(1, v)}
                            whenEntered={() => this.save()}
                        />
                    </Cell>
                </Row>}
                <Row>
                    <Cell flex>
                        <gc.ui.TextArea
                            label="WKT"
                            value={this.toWKT(dd.shape)}
                            whenChanged={v => this.updateFromWKT(v)}
                            height={200}
                        />
                    </Cell>
                </Row>
            </Form>
        </gc.ui.Dialog>;
    }
}
