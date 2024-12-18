import * as React from 'react';

import * as gc from 'gc';

let {Row, Cell} = gc.ui.Layout;

interface ToolboxProps extends gc.types.ViewProps {
    appActiveTool: string;
}

const ToolboxStoreKeys = [
    'appActiveTool',
];

interface ToolboxViewProps extends gc.types.ViewProps {
    iconClass?: string;
    title?: string;
    hint?: string;
    buttons?: Array<React.ReactElement<any>>;
}

export class Content extends gc.View<ToolboxViewProps> {
    render() {
        return <div className="modToolboxContent">
            <Row className="modToolboxContentHeader">
                <Cell>
                    <div className="modToolboxContentTitle">{this.props.title}</div>
                </Cell>
            </Row>
            <Row className="modToolboxContentFooter">
                <Cell flex/>
                {this.props.buttons && this.props.buttons.map((b, n) => <Cell key={n}>{b}</Cell>)}
                <Cell>
                    <gc.ui.Button
                        className="modToolboxCancelButton"
                        tooltip={this.props.controller.__('cmpToolCancelButton')}
                        whenTouched={() => this.props.controller.app.stopTool('')}
                    />
                </Cell>
            </Row>

        </div>;

        /*
        return <div className="modToolboxContent">
            <Row className="modToolboxContentHeader">
                <Cell>
                    <gc.ui.Button
                        className={this.props.iconClass}
                    />
                </Cell>
                <Cell>
                    <div className="modToolboxContentTitle">{this.props.title}</div>
                    <div className="modToolboxContentHint">{this.props.hint}</div>
                </Cell>
            </Row>

            <Row className="modToolboxContentFooter">
                <Cell flex/>
                {this.props.buttons && this.props.buttons.map((b, n) => <Cell key={n}>{b}</Cell>)}
                <Cell>
                    <gc.ui.Button
                        className="modToolboxCancelButton"
                        tooltip={this.props.controller.__('modDrawCancelButton')}
                        whenTouched={() => this.props.controller.app.stopTool('')}
                    />
                </Cell>
            </Row>
        </div>
        */
    }
}

class ToolboxView extends gc.View<ToolboxProps> {
    render() {
        let view = this.props.controller.app.activeTool.toolboxView;

        if (!view)
            return <div className="modToolbox"/>;

        return <div className="modToolbox isActive">{view}</div>;

    }
}

class ToolboxController extends gc.Controller {
    uid: 'Toolbox';

    get appOverlayView() {
        return this.createElement(
            this.connect(ToolboxView, ToolboxStoreKeys));
    }

}

gc.registerTags({
    'Shared.Toolbox': ToolboxController,
});


