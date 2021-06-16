import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

import {Touchable, Button} from '../button';
import {Row, Cell} from '../layout';
import {Error, Info} from '../text';


interface DialogProps {
    className?: string;
    whenClosed?: any;
    whenZoomed?: any;
    quickClose?: boolean;
    title?: string;
    header?: React.ReactElement;
    footer?: React.ReactElement;
    buttons?: Array<React.ReactElement>;
    frame?: string;
    style?: object;
}

class DialogHeader extends base.Pure<DialogProps> {
    render() {
        if (this.props.header)
            return <div className="uiDialogHeader">{this.props.header}</div>;

        if (!this.props.title && !this.props.whenClosed)
            return null;

        let header = <Row>
            {this.props.title && <Cell>
                <div className="uiDialogTitle">{this.props.title}</div>
            </Cell>}
            <Cell flex/>
            {this.props.whenZoomed && <Cell><Touchable
                className='uiDialogZoomButton'
                whenTouched={() => this.props.whenZoomed()}
            /></Cell>}
            {this.props.whenClosed && <Cell><Touchable
                className='uiDialogCloseButton'
                whenTouched={() => this.props.whenClosed()}
            /></Cell>}
        </Row>;

        return <div className="uiDialogHeader">{header}</div>;
    }
}

class DialogFooter extends base.Pure<DialogProps> {
    render() {
        if (this.props.footer)
            return <div className="uiDialogFooter">{this.props.footer}</div>;

        if (!this.props.buttons)
            return null;

        return <div className="uiDialogFooter">
            <Row>
                <Cell flex/>
                {this.props.buttons.map((b, n) => <Cell key={n}>{b}</Cell>)}
            </Row>
        </div>;
    }
}

class DialogContent extends base.Pure<DialogProps> {
    render() {
        if (this.props.frame)
            return <div className="uiDialogFrameContent">
                <iframe src={this.props.frame}/>
            </div>;

        return <div className="uiDialogContent">
            {this.props.children}
        </div>;
    }
}


export class Dialog extends base.Pure<DialogProps> {
    backdropRef: React.RefObject<HTMLDivElement>;

    constructor(props) {
        super(props);
        this.backdropRef = React.createRef();
    }

    render() {
        let cls = util.className('uiDialog', this.props.className);

        return <div
            className='uiDialogBackdrop'
            onClick={e => this.onClose(e)}
            ref={this.backdropRef}
        >
            <div className={cls} style={this.props.style}>
                <DialogHeader {...this.props}/>
                <DialogContent {...this.props}/>
                <DialogFooter {...this.props}/>
            </div>
        </div>
    }

    protected onClose(e: React.MouseEvent) {
        let canClose = this.props.whenClosed
            && this.props.quickClose
            && e.target
            && e.target === this.backdropRef.current;

        if (canClose)
            this.props.whenClosed();
    }

}

//


interface ConfirmProps {
    className?: string;
    whenConfirmed: () => any;
    whenRejected?: () => any;
    quickClose?: boolean;
    title?: string;
    text: string;
    details?: string;

}

export class Confirm extends base.Pure<ConfirmProps> {
    render() {
        let cls = util.className(
            'uiConfirm',
            this.props.className);

        let content = <Info text={this.props.text} details={this.props.details}/>;
        let buttons = [
            <Button
                className="cmpButtonFormOk"
                whenTouched={this.props.whenConfirmed}
                primary
            />,
            <Button
                className="cmpButtonFormCancel"
                whenTouched={this.props.whenRejected}
            />,
        ];

        return <Dialog
            className={cls}
            whenClosed={this.props.whenRejected}
            title={this.props.title}
            buttons={buttons}
            quickClose={this.props.quickClose}
        >{content}</Dialog>
    }
}


//

interface AlertProps {
    className?: string;
    whenClosed?: any;
    title?: string;
    style?: object;
    error?: string;
    info?: string;
    details?: string;
}

export class Alert extends base.Pure<AlertProps> {
    render() {
        let cls = util.className(
            'uiAlert',
            this.props.error && 'uiAlertError',
            this.props.info && 'uiAlertInfo',
            this.props.className);

        let content = null;

        if (this.props.error) {
            content = <Error text={this.props.error} details={this.props.details}/>
        }
        if (this.props.info) {
            content = <Info text={this.props.info} details={this.props.details}/>
        }

        return <Dialog
            className={cls}
            whenClosed={this.props.whenClosed}
            title={this.props.title}
            style={this.props.style}
            quickClose
        >{content}</Dialog>
    }
}


//

export class Popup extends base.Pure<DialogProps> {
    backdropRef: React.RefObject<HTMLDivElement>;

    constructor(props) {
        super(props);
        this.backdropRef = React.createRef();
    }

    componentDidMount() {
        util.nextTick(() => this.backdropRef.current && (this.backdropRef.current.className += ' isActive'));
    }

    render() {
        return <div
            className='uiPopupBackdrop'
            onClick={e => this.onClose(e)}
            ref={this.backdropRef}
        >
            <div
                className={util.className('uiPopup', this.props.className)}
                style={this.props.style}
            >{this.props.children}</div>
        </div>
    }

    protected onClose(e: React.MouseEvent) {
        let ok = this.props.whenClosed
            && e.target
            && e.target === this.backdropRef.current;

        if (ok)
            this.props.whenClosed();
    }
}

//

export class Panel extends base.Pure<DialogProps> {
    render() {
        let cls = util.className(
            'uiPanel',
            this.props.whenClosed && 'withCloseButton',
            this.props.className);

        return <div className={cls}>
            {this.props.whenClosed && <Touchable
                className='uiPanelCloseButton'
                whenTouched={() => this.props.whenClosed()}
            />}
            <div className='uiPanelContent'>
                {this.props.children}
            </div>
        </div>
    }
}
