import * as React from 'react';

import * as gws from 'gws';

let {Row, Cell} = gws.ui.Layout;

export class Infobox extends React.PureComponent<gws.types.ViewProps> {

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
                        <gws.ui.Button
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
