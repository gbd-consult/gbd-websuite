import * as React from 'react';

import * as gws from 'gws';

import * as types from './types';
import type {Controller} from './controller';
import {FormTab} from './form_tab';
import {ListTab} from './list_tab';
import {ModelsTab} from './models_tab';


export abstract class SidebarView extends gws.View<types.ViewProps> {
    abstract master(): Controller;

    render() {
        let es = this.master().editState;

        if (es.isWaiting)
            return <gws.ui.Loader/>;

        if (es.sidebarSelectedFeature)
            return <FormTab {...this.props} controller={this.master()} />;

        if (es.sidebarSelectedModel)
            return <ListTab {...this.props} controller={this.master()} />;

        return <ModelsTab {...this.props} controller={this.master()}  />;
    }
}
