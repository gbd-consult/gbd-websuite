import * as React from 'react';

import * as util from './util';


export class Pure<P = {}> extends React.PureComponent<P> {
}

export class Component<P, S> extends React.Component<P, S> {
    state: S = {} as S;
}

//

export interface ControlProps {
    className?: string;
    disabled?: boolean;
    label?: string;
    placeholder?: string;
    dropUp?: boolean;
    tooltip?: string;
    focusRef?: React.RefObject<any>;

}

export interface ControlState {
    hasFocus: boolean;
    isOpen: boolean;
}

const FOCUS_WAIT_TIMEOUT = 200;

export class Control<P extends ControlProps = ControlProps, S extends ControlState = ControlState> extends Component<P, S> {
    protected _blurTimer: any = 0;
    protected _focusRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.state.hasFocus = false;
        this.state.isOpen = false;
        this._blurTimer = 0;
        this._focusRef = React.createRef();
    }

    componentWillUnmount() {
        clearTimeout(this._blurTimer);
    }

    protected get focusRef() {
        return this.props.focusRef || this._focusRef;

    }

    protected whenFocusChanged(on) {
    }

    protected grabFocus() {
        if (this.focusRef && this.focusRef.current) {
            this.focusRef.current.focus();
        }
    }

    onFocus(e) {
        clearTimeout(this._blurTimer);
        this.setFocus(true);
    }

    onBlur(e) {
        clearTimeout(this._blurTimer);
        this._blurTimer = setTimeout(() => this.setFocus(false), FOCUS_WAIT_TIMEOUT);
    }

    protected setFocus(on) {
        if (on && !this.state.hasFocus) {
            return this.setState({hasFocus: true}, () => this.whenFocusChanged(true));
        }
        if (!on && this.state.hasFocus)
            return this.setState({hasFocus: false, isOpen: false}, () => this.whenFocusChanged(false))
    }

    protected setOpen(isOpen) {
        this.setState({isOpen});
    }

    protected toggleOpen() {
        this.setOpen(!this.state.isOpen);
    }
}


//

export interface InputProps<T> extends ControlProps {
    value: T;
    readOnly?: boolean;
    whenChanged?: (value?: T) => void;
    whenEntered?: (value?: T) => void;
}

//

/*

The anatomy of a Control

+-------------------------------------------+
|  content uiControl.uiXXX (flex)           |
|                                           |
|     +---------------------------------+   |
|     | label                           |   |
|     +---------------------------------+   |
|                                           |
|     +---------------------------------+   |
|     | body (relative)                 |   |
|     |                                 |   |
|     |    +------------------------+   |   |
|     |    | box (flex)             |   |   |
|     |    |                        |   |   |
|     |    |   +----------------+   |   |   |
|     |    |   | input, etc     |   |   |   |
|     |    |   +----------------+   |   |   |
|     |    |                        |   |   |
|     |    |   +----------------+   |   |   |
|     |    |   | aux button     |   |   |   |
|     |    |   +----------------+   |   |   |
|     |    |                        |   |   |
|     |    +------------------------+   |   |
|     |                                 |   |
|     |    +------------------------+   |   |
|     |    | dropdown (abs)         |   |   |
|     |    +------------------------+   |   |
|     |                                 |   |
|     +---------------------------------+   |
|                                           |
+-------------------------------------------+

*/

interface ContentProps {
    withClass?: string;
    withProps?: object;
    withRef?: React.Ref<HTMLDivElement>;
    noLabel?: boolean;
    of: Control;
}

export class Content extends Pure<ContentProps> {
    render() {
        let cc = this.props.of,
            props = {
                className: util.className(
                    'uiControl',
                    this.props.withClass,
                    cc.props.className,
                    cc.state.hasFocus && 'hasFocus',
                    cc.props.disabled && 'isDisabled',
                    cc.props.dropUp && 'isDropUp',
                    cc.state.isOpen && 'isOpen'
                ),
                onFocus: e => cc.onFocus(e),
                onBlur: e => cc.onBlur(e),
                ...(this.props.withProps || {})
            };

        return <div {...props} ref={this.props.withRef}>
            {cc.props.label && !this.props.noLabel && <div className="uiLabel">{cc.props.label}</div>}
            <div className="uiControlBody">
                {this.props.children}
            </div>
        </div>;
    }
}

interface BoxProps {
    withProps?: object;
    withRef?: React.Ref<HTMLDivElement>;
}

export class Box extends Pure<BoxProps> {
    render() {
        return <div className="uiControlBox" ref={this.props.withRef} {...this.props.withProps}>
            {this.props.children}
        </div>;
    }
}

interface DropDownProps {
    withProps?: object;
    withRef?: React.Ref<HTMLDivElement>;
}

export class DropDown extends Pure<DropDownProps> {
    render() {
        return <div className="uiDropDown" ref={this.props.withRef} {...this.props.withProps}>
            {this.props.children}
        </div>;
    }
}
