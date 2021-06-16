import * as React from 'react';

import * as types from '../types';
import * as ui from '../ui';

let {Row, Cell} = ui.Layout;

export class Infobox extends React.PureComponent<types.ViewProps> {

    render() {
        let cc = this.props.controller;

        let close = () => cc.update({
            marker: null,
            infoboxContent: null
        });


        return <div className="cmpInfoboxContent">
            <div className="cmpInfoboxBody">
                {this.props.children}
            </div>
            <div className="cmpInfoboxFooter">
                <Row>
                    <Cell flex/>
                    <Cell>
                        <ui.Button
                            className='cmpInfoboxCloseButton'
                            tooltip={cc.__('cmpInfoboxCloseButton')}
                            whenTouched={close}
                        />
                    </Cell>
                </Row>
            </div>
        </div>
    }
}
