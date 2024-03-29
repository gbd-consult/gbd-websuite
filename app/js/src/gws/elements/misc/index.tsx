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

    updateVisible() {
        let v = this.getValue('appMediaWidth');
        this.update({
            altbarVisible: (v === 'large' || v === 'xlarge')
        })
    }

    async init() {
        await super.init();
        this.updateVisible();
        this.app.whenChanged('appMediaWidth', v => this.updateVisible());
    }
}

interface InfoboxProps extends gws.types.ViewProps {
    controller: InfoboxController;
    infoboxContent: React.ReactElement<any>;

}

class InfoboxView extends gws.View<InfoboxProps> {
    render() {
        return <div {...gws.lib.cls('cmpInfobox', this.props.infoboxContent && 'isActive')}>
            {this.props.infoboxContent || null}
        </div>;
    }
}

class InfoboxController extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(InfoboxView, ['infoboxContent'])
        )
    }
}

interface DialogProps extends gws.types.ViewProps {
    controller: DialogController;
    dialogContent: any;

}

class DialogView extends gws.View<DialogProps> {
    render() {
        if (!this.props.dialogContent)
            return null;
        return <gws.ui.Dialog
            {...this.props.dialogContent}
            whenClosed={_ => this.props.controller.update({dialogContent: null})}
        />;
    }
}

class DialogController extends gws.Controller {
    get appOverlayView() {
        return this.createElement(
            this.connect(DialogView, ['dialogContent'])
        )
    }
}

gws.registerTags({
    'Shared.Dialog': DialogController,
    'Shared.Infobox': InfoboxController,
    'Altbar': AltbarController,

});

