import * as React from 'react';

import * as gc from 'gc';

import * as types from './types';
import type {Controller} from './controller';
import {FormTab} from './form_tab';
import {ListTab} from './list_tab';
import {ModelsTab} from './models_tab';


export abstract class SidebarView extends gc.View<types.ViewProps> {
    abstract master(): Controller;

    render() {
        let es = this.master().editState;

        if (es.isWaiting)
            return <gc.ui.Loader/>;

        if (es.sidebarSelectedFeature)
            return <FormTab {...this.props} controller={this.master()} />;

        if (es.sidebarSelectedModel)
            return <ListTab {...this.props} controller={this.master()} />;

        return <ModelsTab {...this.props} controller={this.master()}  />;
    }
}
