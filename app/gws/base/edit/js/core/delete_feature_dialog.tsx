import * as React from 'react';

import * as gws from 'gws';
import * as types from './types';
import type {Controller} from './controller';

export class DeleteFeatureDialog extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    render() {
        let cc = this.master();
        let dd = cc.editState.dialogData as types.DeleteFeatureDialogData;

        return <gws.ui.Confirm
            title={this.__('editDeleteFeatureTitle')}
            text={this.__('editDeleteFeatureText').replace(/%s/, dd.feature.views.title)}
            whenConfirmed={() => dd.whenConfirmed()}
            whenRejected={() => cc.closeDialog()}
        />
    }
}

