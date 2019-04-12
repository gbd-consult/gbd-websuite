import * as React from 'react';
import * as gws from 'gws';

let {Row, Cell} = gws.ui.Layout;

interface PositionProps extends gws.types.ViewProps {
    mapPointerX: number;
    mapPointerY: number;
}

class PositionView extends gws.View<PositionProps> {
    render() {
        return <div className="modInfobarWidget modInfobarPosition">
            <Cell className="modInfobarLabel">{this.__('modInfobarPosition')}</Cell>
            <Cell className="modInfobarPositionInput">
                <gws.ui.TextInput
                    value={this.props.mapPointerX + ', ' + this.props.mapPointerY}
                    disabled
                />
            </Cell>
        </div>
    }
}

class PositionWidget extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(PositionView, ['mapPointerX', 'mapPointerY']));
    }

}

interface LoaderProps extends gws.types.ViewProps {
    appRequestCount: number;
}

const LOADER_PROGRESS_MIN = 5;
const LOADER_PROGRESS_MAX = 50;

class LoaderView extends gws.View<LoaderProps> {
    render() {
        let rc = this.props.appRequestCount;
        let bars = Math.min(rc, LOADER_PROGRESS_MAX);

        return <div {...gws.tools.cls('modInfobarWidget', 'modInfobarLoader', rc && 'isActive')}>
            <Row>
                <Cell>
                    <gws.ui.IconButton  className='modInfobarLoaderIcon'/>
                </Cell>
                {rc > LOADER_PROGRESS_MIN && <Cell width={bars * 2}>
                    {gws.tools.range(bars).map(n => <div className='modInfobarLoaderBar' key={n}/>)}
                </Cell>}
            </Row>
        </div>
    }
}

class LoaderWidget extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(LoaderView, ['appRequestCount']));
    }

}

interface ScaleProps extends gws.types.ViewProps {
    controller: ScaleWidget;
    mapEditScale: number;
}

class ScaleView extends gws.View<ScaleProps> {
    render() {
        let cc = this.props.controller;

        let n = gws.tools.asNumber(this.props.mapEditScale),
            res = this.props.controller.map.resolutions,
            max = Math.max(...res.map(gws.tools.res2scale)),
            min = Math.min(...res.map(gws.tools.res2scale));

        return <div className="modInfobarWidget modInfobarScale">
            <Cell className="modInfobarLabel">{this.__('modInfobarScale')}</Cell>
            <Cell className="modInfobarScaleSlider">
                <gws.ui.Slider
                    minValue={min}
                    maxValue={max}
                    value={n}
                    whenChanged={v => cc.setValue(v, true)}
                    whenInteractionStarted={() => cc.map.setInteracting(true)}
                    whenInteractionStopped={() => cc.map.setInteracting(false)}
                />
            </Cell>
            <Cell className="modInfobarScaleInput">
                <gws.ui.TextInput
                    value={String(n)}
                    whenChanged={v => cc.setValue(v, false)}
                    whenEntered={v => cc.setValue(v, true)}
                />
            </Cell>
        </div>
    }
}

class ScaleWidget extends gws.Controller {
    async init() {
        this.app.whenChanged('mapScale', s => this.update({'mapEditScale': s}))
    }

    setValue(value, withMap) {
        value = Number(value) || 1;
        if (withMap)
            this.map.setScale(value);
        this.update({'mapEditScale': value})
    }

    get defaultView() {
        return this.createElement(
            this.connect(ScaleView,
                ['mapEditScale']));
    }
}

interface RotationProps extends gws.types.ViewProps {
    controller: RotationWidget;
    mapEditAngle: number;
}

class RotationView extends gws.View<RotationProps> {
    update(value) {
        this.props.controller.map.setAngle(value);
    }

    render() {
        let cc = this.props.controller;
        let n = gws.tools.asNumber(this.props.mapEditAngle);

        return <div className="modInfobarWidget modInfobarRotation">
            <Cell className="modInfobarLabel">{this.__('modInfobarRotation')}</Cell>
            <Cell className="modInfobarRotationSlider">
                <gws.ui.Slider
                    minValue={0}
                    maxValue={360}
                    value={n}
                    whenChanged={v => cc.setValue(v, true)}
                    whenInteractionStarted={() => cc.map.setInteracting(true)}
                    whenInteractionStopped={() => cc.map.setInteracting(false)}
                />
            </Cell>
            <Cell className="modInfobarRotationInput">
                <gws.ui.TextInput
                    value={String(n)}
                    whenChanged={v => cc.setValue(v, false)}
                    whenEntered={v => cc.setValue(v, true)}
                />
            </Cell>
        </div>
    }
}

class RotationWidget extends gws.Controller {
    async init() {
        this.app.whenChanged('mapAngle', s => this.update({'mapEditAngle': s}))
    }

    setValue(value, withMap) {
        value = Number(value) || 0;
        if (withMap)
            this.map.setAngle(value);
        this.update({'mapEditAngle': value})
    }

    get defaultView() {
        return this.createElement(
            this.connect(RotationView,
                ['mapEditAngle']));
    }
}

class Spacer extends gws.Controller {

    get defaultView() {
        return <Cell flex/>;
    }
}

interface LinkProps extends gws.types.ViewProps {
    controller: LinkWidget;
    title: string;
    href: string;
    className?: string;
}

class LinkView extends gws.View<LinkProps> {
    render() {
        return <div {...gws.tools.cls('modInfobarWidget', 'modInfobarLink', this.props.className)}>
            <a
                onClick={() => this.props.controller.touched()}>{this.props.title}</a>
        </div>
    }
}

class LinkButtonView extends gws.View<LinkProps> {
    render() {
        return <gws.ui.IconButton
            className={this.props.className}
            tooltip={this.props.title}
            whenTouched={() => this.props.controller.touched()}
        />
    }
}

class LinkWidget extends gws.Controller {

    touched() {
        let url = this.options.href;

        if (this.options.target === 'frame')
            this.update({dialogContent: <iframe src={url}/>});
        else if (this.options.target === 'blank')
            window.open(url);
        else
            location.href = url;
    }

    get defaultView() {
        return this.createElement(LinkView, this.options);
    }

}

class HelpWidget extends LinkWidget {

    get defaultView() {
        this.options = {
            ...this.options,
            target: 'frame',
            href: this.getValue('helpUrl'),
            title: this.__('modInfobarHelpTitle'),
            className: 'modInfobarHelpButton',
        };
        return this.createElement(LinkButtonView, this.options);
    }
}

class HomeLinkWidget extends LinkWidget {

    get defaultView() {
        this.options = {
            ...this.options,
            href: this.getValue('homeUrl'),
            title: this.__('modInfobarHomeLinkTitle'),
            className: 'modInfobarHomeLinkButton',
        };
        return this.createElement(LinkButtonView, this.options);
    }
}

const ABOUT_URL = 'https://gws.gbd-consult.de/';

class AboutWidget extends LinkWidget {

    get defaultView() {
        this.options = {
            ...this.options,
            href: ABOUT_URL,
            target: 'blank',
            title: this.__('modInfobarAboutTitle'),
            className: 'modInfobarAboutButton',
        };
        return this.createElement(LinkButtonView, this.options);
    }
}

// class LeftSide extends gws.Controller {
//
//     get defaultView() {
//         return <React.Fragment>
//             {this.renderChildren()}
//             <Cell flex/>
//         </React.Fragment>
//     }
// }
//
// class RightSide extends gws.Controller {
//
//     get defaultView() {
//         return this.renderChildren();
//     }
// }

class InfobarController extends gws.Controller {

    get defaultView() {
        return <div className="modInfobar">
            {this.renderChildren()}
        </div>;
    }
}

export const tags = {
    'Infobar': InfobarController,
    // 'Infobar.LeftSide': LeftSide,
    // 'Infobar.RightSide': RightSide,
    'Infobar.Link': LinkWidget,
    'Infobar.Help': HelpWidget,
    'Infobar.About': AboutWidget,
    'Infobar.Position': PositionWidget,
    'Infobar.Rotation': RotationWidget,
    'Infobar.Scale': ScaleWidget,
    'Infobar.HomeLink': HomeLinkWidget,
    'Infobar.Loader': LoaderWidget,
    'Infobar.Spacer': Spacer,
};
