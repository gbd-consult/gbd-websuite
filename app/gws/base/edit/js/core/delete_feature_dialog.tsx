import * as React from 'react';

import * as gws from 'gws';
import * as types from './types';
import type {Controller} from './controller';

export class DeleteFeatureDialog extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    render() {
        let dd = this.props.editDialogData as types.DeleteFeatureDialogData;
        let cc = this.master();

        return <gws.ui.Confirm
            title={this.__('editDeleteFeatureTitle')}
            text={this.__('editDeleteFeatureText').replace(/%s/, dd.feature.views.title)}
            whenConfirmed={() => dd.whenConfirmed()}
            whenRejected={() => cc.closeDialog()}
        />
    }
}

