import * as React from 'react';

import * as gws from 'gws';

interface DrawControlsProps extends gws.types.ViewProps {
    mapDrawEndFunction: any;
}

class DrawControlsView extends gws.View<DrawControlsProps> {

    render() {
        return <div {...gws.tools.cls('modDrawControls', this.props.mapDrawEndFunction && 'isActive')}>
            <gws.ui.IconButton
                className="modToolbarDrawOk"
                tooltip={this.__('modToolbarDrawOk')}
                whenTouched={() => this.props.mapDrawEndFunction(true)}
            />
            <gws.ui.IconButton
                className="modToolbarDrawCancel"
                tooltip={this.__('modToolbarDrawCancel')}
                whenTouched={() => this.props.mapDrawEndFunction(false)}
            />
        </div>
    }

}

class DrawControls extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(DrawControlsView, ['mapDrawEndFunction'])
        );
    }
}

export const tags = {
    'Shared.DrawControls': DrawControls,
};


