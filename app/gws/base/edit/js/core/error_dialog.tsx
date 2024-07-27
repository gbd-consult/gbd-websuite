import * as React from 'react';

import * as gws from 'gws';
import * as types from './types';
import type {Controller} from './controller';

export class ErrorDialog extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }
    render() {
        let dd = this.props.editDialogData as types.ErrorDialogData;
        let cc = this.master();

        return <gws.ui.Alert
            title={'Fehler'}
            error={dd.errorText}
            details={dd.errorDetails}
            whenClosed={() => cc.closeDialog()}
        />
    }
}

