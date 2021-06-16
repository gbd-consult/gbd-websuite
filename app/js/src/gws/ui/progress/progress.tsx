import * as React from 'react';

import * as base from './base';
import * as util from './util';


interface ProgressProps extends base.ControlProps {
    value: number;
}

export class Progress extends base.Control<ProgressProps> {
    render() {
        let style = {
            width: Math.floor(util.constrain(this.props.value, 0, 100)) + '%'
        };
        return <base.Content of={this} withClass="uiProgress">
            <base.Box>
                <div className='uiBackgroundBar'/>
                <div className='uiActiveBar' style={style}/>
            </base.Box>
        </base.Content>
    }
}

export class Loader extends base.Pure {
    render() {
        return <div className='uiLoader'/>;
    }
}

