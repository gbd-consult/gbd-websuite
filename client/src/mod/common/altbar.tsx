import * as React from 'react';

import * as gws from 'gws';

interface AltbarProps extends gws.types.ViewProps {
    altbarVisible: boolean;
}

class AltbarView extends gws.View<AltbarProps> {
    render() {
        if (!this.props.altbarVisible)
            return null;

        return <div className="modAltbar">
            {this.props.controller.renderChildren()}
        </div>
    }
}

class AltbarController extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(AltbarView, ['altbarVisible']));
    }

    async init() {
        this.app.whenChanged('appMediaWidth', v =>
            this.update({
                altbarVisible: (v === 'large' || v === 'xlarge')
            })
        );

    }
}

export const tags = {
    'Altbar': AltbarController,
};

