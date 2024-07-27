import * as React from 'react';

import * as gws from 'gws';

import * as types from './types';

import {TableViewDialog} from './table_view_dialog';
import {SelectModelDialog} from './select_model_dialog';
import {SelectFeatureDialog} from './select_feature_dialog';
import {DeleteFeatureDialog} from './delete_feature_dialog';
import {GeometryTextDialog} from './geometry_text_dialog';
import {ErrorDialog} from './error_dialog';


export class Dialog extends gws.View<types.ViewProps> {

    render() {
        let model = this.props.editState.tableViewSelectedModel;
        if (model) {
            return <TableViewDialog {...this.props} />
        }

        let dd = this.props.editDialogData;
        if (!dd || !dd.type) {
            return null;
        }

        return this.renderDialogByType(dd.type);
    }

    renderDialogByType(type: string) {
        switch (type) {
            case 'SelectModel':
                return <SelectModelDialog {...this.props} />
            case 'SelectFeature':
                return <SelectFeatureDialog {...this.props} />
            case 'DeleteFeature':
                return <DeleteFeatureDialog {...this.props} />
            case 'Error':
                return <ErrorDialog {...this.props} />
            case 'GeometryText':
                return <GeometryTextDialog {...this.props} />
        }
    }
}
