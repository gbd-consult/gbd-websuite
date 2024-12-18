import * as React from 'react';

import * as gc from 'gc';

let {Row, Cell} = gc.ui.Layout;

export class Infobox extends React.PureComponent<gc.types.ViewProps> {

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
                        <gc.ui.Button
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
