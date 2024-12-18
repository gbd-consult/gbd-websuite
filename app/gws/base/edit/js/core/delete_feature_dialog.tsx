import * as React from 'react';

import * as gc from 'gc';
import * as types from './types';
import type {Controller} from './controller';

export class DeleteFeatureDialog extends gc.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    render() {
        let cc = this.master();
        let dd = cc.editState.dialogData as types.DeleteFeatureDialogData;

        return <gc.ui.Confirm
            title={this.__('editDeleteFeatureTitle')}
            text={this.__('editDeleteFeatureText').replace(/%s/, dd.feature.views.title)}
            whenConfirmed={() => dd.whenConfirmed()}
            whenRejected={() => cc.closeDialog()}
        />
    }
}

