import * as React from 'react';

import * as gc from 'gc';

interface AltbarProps extends gc.types.ViewProps {
    altbarVisible: boolean;
}

class AltbarView extends gc.View<AltbarProps> {
    render() {
        if (!this.props.altbarVisible)
            return null;

        return <div className="modAltbar">
            {this.props.controller.renderChildren()}
        </div>
    }
}

class AltbarController extends gc.Controller {
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

interface InfoboxProps extends gc.types.ViewProps {
    controller: InfoboxController;
    infoboxContent: React.ReactElement<any>;

}

class InfoboxView extends gc.View<InfoboxProps> {
    render() {
        return <div {...gc.lib.cls('cmpInfobox', this.props.infoboxContent && 'isActive')}>
            {this.props.infoboxContent || null}
        </div>;
    }
}

class InfoboxController extends gc.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(InfoboxView, ['infoboxContent'])
        )
    }
}

interface DialogProps extends gc.types.ViewProps {
    controller: DialogController;
    dialogContent: any;

}

class DialogView extends gc.View<DialogProps> {
    render() {
        if (!this.props.dialogContent)
            return null;
        return <gc.ui.Dialog
            {...this.props.dialogContent}
            whenClosed={_ => this.props.controller.update({dialogContent: null})}
        />;
    }
}

class DialogController extends gc.Controller {
    get appOverlayView() {
        return this.createElement(
            this.connect(DialogView, ['dialogContent'])
        )
    }
}

gc.registerTags({
    'Shared.Dialog': DialogController,
    'Shared.Infobox': InfoboxController,
    'Altbar': AltbarController,

});

