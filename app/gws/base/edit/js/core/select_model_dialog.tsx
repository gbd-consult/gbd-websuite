import * as React from 'react';

import * as gc from 'gc';
import * as types from './types';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gc.ui.Layout;


export class SelectModelDialog extends gc.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    render() {
        let cc = this.master();
        let dd = cc.editState.dialogData as types.SelectModelDialogData;

        let cancelButton = <gc.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        let items = dd.models.map(model => ({
            value: model.uid,
            text: model.title,
        }));

        return <gc.ui.Dialog
            className="editSelectModelDialog"
            title={this.__('editSelectModelTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[cancelButton]}
        >
            <Form>
                <Row>
                    <Cell flex>
                        <gc.ui.List
                            items={items}
                            value={null}
                            whenChanged={v => dd.whenSelected(cc.app.modelRegistry.getModel(v))}
                        />
                    </Cell>
                </Row>
            </Form>
        </gc.ui.Dialog>;
    }
}

