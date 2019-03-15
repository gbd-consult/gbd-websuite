import * as React from 'react';
import * as types from '../types';


export class View<P extends types.ViewProps> extends React.PureComponent<P> {
    get app() {
        return this.props.controller.app;
    }

    __(key) {
        return this.props.controller.__(key);
    }
}
