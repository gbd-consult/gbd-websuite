import * as React from 'react';

import * as gws from 'gws';

interface PopupProps extends gws.types.ViewProps {
    controller: PopupController;
    popupContent: React.ReactElement<any>;

}

class PopupView extends gws.View<PopupProps> {
    render() {
        return <div {...gws.tools.cls('modPopup', this.props.popupContent && 'isActive')}>
            {this.props.popupContent || null}
        </div>;
    }
}

class PopupController extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(PopupView, ['popupContent'])
        )
    }
}

interface DialogProps extends gws.types.ViewProps {
    controller: DialogController;
    dialogContent: React.ReactElement<any>;

}

class DialogView extends gws.View<DialogProps> {
    render() {
        if (!this.props.dialogContent)
            return null;
        return <gws.ui.Dialog whenClosed={_ => this.props.controller.update({dialogContent: null})}>
            {this.props.dialogContent}
        </gws.ui.Dialog>;
    }
}

class DialogController extends gws.Controller {
    get appOverlayView() {
        return this.createElement(
            this.connect(DialogView, ['dialogContent'])
        )
    }
}

export const tags = {
    'Shared.Dialog': DialogController,
    'Shared.Popup': PopupController,
};

