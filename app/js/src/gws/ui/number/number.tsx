import * as React from 'react';

import * as base from './base';
import * as util from './util';

import {Touchable} from './button';
import {Tracker} from './tracker';


interface NumberInputProps extends base.InputProps<number> {
    minValue?: number;
    maxValue?: number;
    step?: number;
    withClear?: boolean;
    locale?: util.Locale;
}


interface NumberInputState extends base.ControlState {
    strValue: string;
    numValue: number;
}

const STEP_INTERVAL = 200;


export class NumberInputBox extends base.Control<NumberInputProps, NumberInputState> {
    constructor(props) {
        super(props);
        this.state = {
            ...this.state,
            ...stateFromNumber(props.value, props),
        }
    }

    static getDerivedStateFromProps(props, state) {
        if (props.value === state.numValue || (Number.isNaN(props.value) && Number.isNaN(state.numValue))) {
            return state;
        }
        return {
            ...state,
            ...stateFromNumber(props.value, props),
        }
    }

    render() {
        let inputProps = {
            className: 'uiRawInput',
            disabled: this.props.disabled,
            onChange: e => this.onChange(e),
            onKeyDown: e => this.onKeyDown(e),
            placeholder: this.props.placeholder || '',
            ref: this.focusRef,
            tabIndex: 0,
            title: this.props.tooltip || '',
            type: 'text',
            value: this.state.strValue || '',
        };


        return <base.Box>
            <input {...inputProps}/>

            {this.props.withClear && <Touchable
                className={util.className('uiClearButton', util.empty(this.props.value) && 'isHidden')}
                whenTouched={() => this.clear()}/>
            }

            {this.props.step && <Touchable className='uiNumberUpDownButton'>
                <Tracker
                    yValueMin={-1}
                    yValueMax={+1}
                    yValue={0}
                    noHandle
                    noMove
                    whenChanged={(x, y) => this.trackDelta(-util.sign(y))}
                    idleInterval={STEP_INTERVAL}
                />
            </Touchable>}
        </base.Box>
    }


    protected onChange(e: React.SyntheticEvent<HTMLInputElement>) {
        let s = stateFromString(e.currentTarget.value, this.props);
        this.setState(s);
        if (this.props.whenChanged)
            this.props.whenChanged(s.numValue);

    }

    protected onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key === 'Enter' && this.props.whenEntered) {
            let s = stateFromString(e.currentTarget.value, this.props);
            this.setState(s);
            this.props.whenEntered(s.numValue);
        }
    }

    protected clear() {
        this.grabFocus();
        let s = stateFromString('', this.props);
        this.setState(s);
        if (this.props.whenChanged)
            this.props.whenChanged(s.numValue);
    }

    protected trackDelta(d) {
        this.grabFocus();

        let n = this.state.numValue;

        n = util.align(n, this.props.step || 1, d);

        n = util.constrain(n,
            this.props.minValue || 0,
            this.props.maxValue || Infinity);

        let s = stateFromNumber(n, this.props);
        this.setState(s);
        if (this.props.whenChanged)
            this.props.whenChanged(s.numValue);

    }
}

export class NumberInput extends base.Control<NumberInputProps, NumberInputState> {
    render() {
        return <base.Content of={this} withClass="uiInput">
            <NumberInputBox {...this.props}/>
        </base.Content>
    }
}


//

interface SliderProps extends base.InputProps<number> {
    minValue: number;
    maxValue: number;
    step?: number;
    whenInteractionStarted?: () => void;
    whenInteractionStopped?: () => void;
}


interface SliderState extends base.ControlState {
    isDown: boolean;
}

export class Slider extends base.Control<SliderProps, SliderState> {
    boxRef: React.RefObject<any>;
    barRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.state.isDown = false;
        this.boxRef = React.createRef();
        this.barRef = React.createRef();
    }

    render() {
        return <base.Content of={this} withClass="uiSlider">
            <base.Box withRef={this.boxRef}>
                <div className='uiBackgroundBar'/>
                <div className='uiActiveBar' ref={this.barRef}/>
                <Tracker
                    xValueMin={this.props.minValue}
                    xValueMax={this.props.maxValue}
                    yValueMin={0}
                    yValueMax={0}
                    xValue={this.props.value}
                    yValue={0}
                    whenPressed={() => this.whenPressed()}
                    whenReleased={() => this.whenReleased()}
                    whenChanged={(x, y) => this.whenChanged(x, y)}
                />
            </base.Box>
        </base.Content>
    }

    componentDidMount() {
        this.updateBars(this.props.value);
    }

    componentDidUpdate() {
        this.updateBars(this.props.value);
    }


    protected whenPressed() {
        this.setState({isDown: true});
        if (this.props.whenInteractionStarted)
            this.props.whenInteractionStarted();
    }

    protected whenReleased() {
        this.setState({isDown: false});
        if (this.props.whenInteractionStopped)
            this.props.whenInteractionStopped();
    }

    protected whenChanged(x, y) {
        let val = util.constrain(
            util.align(x, this.props.step),
            this.props.minValue,
            this.props.maxValue);
        this.props.whenChanged(val);
        this.updateBars(val);
    }

    protected updateBars(val) {
        let p = util.translate(val, this.props.minValue, this.props.maxValue, 0, this.boxRef.current.offsetWidth);
        this.barRef.current.style.width = p + 'px';

    }
}

//


function stateFromNumber(n: number, props: NumberInputProps) {
    return {
        strValue: util.formatNumber(n, props.locale),
        numValue: n,
    }
}

function stateFromString(val: string, props: NumberInputProps) {
    val = val.trim();

    let sn = '',
        sg = '',
        lo = props.locale,
        hasDec = false;

    if (val[0] === '-') {
        if (props.minValue < 0) {
            sn += '-';
            sg += '-'
        }
        val = val.slice(1);
    }

    for (let c of val) {
        if (/\d/.test(c)) {
            sn += c;
            sg += c;
        } else if (!hasDec) {
            if (lo && c === lo.numberGroup) {
                sg += c;
            }
            else if (lo && c === lo.numberDecimal) {
                sn += '.';
                sg += c;
                hasDec = true;
            }
        }
    }

    return {
        strValue: sg,
        numValue: Number(sn),
    }
}

