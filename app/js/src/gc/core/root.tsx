import * as React from 'react';

import { ViewProps } from '../types';
import { View } from './view';
import { Controller } from './controller';

interface RootViewProps extends ViewProps {
    controller: RootController;
}

class RootView extends View<RootViewProps> {
    render() {
        return this.props.controller.app.store.wrap(
            this.props.controller.appView
        )
    }
}

interface AppViewProps extends ViewProps {
    controller: RootController;
    appExtraViewState: string;
    whenMounted: () => void;
}

class AppView extends View<AppViewProps> {
    componentDidMount() {
        this.props.whenMounted();
    }

    getViews(kind: string): Array<React.ReactNode> {
        let vs = [];

        for (let cc of this.props.controller.children) {
            let view = cc[kind];
            if (view) {
                vs.push(<React.Fragment key={cc.uid + '.' + kind}>{view}</React.Fragment>);
            }
        }
        return vs;
    }

    render() {

        let extraViews = this.getViews('extraView');

        let extraCls = '';
        if (this.props.appExtraViewState === 'max') {
            extraCls = 'withExtraMax'
        }
        else if (this.props.appExtraViewState === 'min') {
            extraCls = 'withExtraMin'
        } else if (extraViews.length > 0) {
            extraCls = 'withExtra';
        }

        return <div>
            <div className={'gwsMapArea ' + extraCls}>
                <div className="gwsMap" />
                <div className="gwsMapOverlay">{this.getViews('mapOverlayView')}</div>
                {this.getViews('defaultView')}
            </div>
            {extraViews.length > 0 && <div className={'gwsExtraArea ' + extraCls}>
                {extraViews}
            </div>}
            <div className="gwsAppOverlay">{this.getViews('appOverlayView')}</div>
        </div>
    }
}


export class RootController extends Controller {
    appClasses = [];

    get appView() {
        return this.createElement(this.connect(AppView, ['appExtraViewState']), {
            children: this.children,
            whenMounted: () => this.app.mounted()
        });
    }

    get defaultView() {
        return this.createElement(RootView, { children: this.children });
    }

    async init() {
        await super.init();
        this.appClasses = this.app.domNode.className.split(' ');

        this.app.whenChanged('altbarVisible', v => this.setClass(v, 'withAltbar'));
        this.app.whenChanged('printerState', v => this.setClass(v, 'withPrintPreview'));
        this.app.whenChanged('appActiveTool', v => this.setClass(v !== 'Tool.Default', 'withToolbox'));

        this.app.whenChanged('appExtraViewState', v => {
            let node = this.app.domNode.querySelector('.gwsMap');
            this.app.map['setTargetDomNode'](node);
        });
    }

    protected setClass(yes, cls) {
        this.appClasses = this.appClasses.filter(x => x !== cls);
        if (yes)
            this.appClasses.push(cls);
        this.app.domNode.className = this.appClasses.join(' ');
    }


}
