import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    whenNewButtonTouched?: () => void;
    whenEditButtonTouched?: () => void;
}

class FormView extends gws.View<Props> {
    render() {
        let cc = this.props.controller;
        let field = this.props.field;
        let hasGeom = Boolean(this.props.feature.geometry);
        let isDrawing = cc.app.activeTool.tag.includes('Draw');

        return <gws.ui.Row>
            {!hasGeom && this.props.whenNewButtonTouched && <gws.ui.Cell>
                <gws.ui.Button
                    {...gws.lib.cls('cmpFormDrawGeometryButton', isDrawing && 'isActive')}
                    tooltip={this.__('widgetGeometryNew')}
                    whenTouched={this.props.whenNewButtonTouched}
                />
            </gws.ui.Cell>}
            {hasGeom && this.props.whenEditButtonTouched && <gws.ui.Cell>
                <gws.ui.Button
                    className='cmpFormEditGeometryButton'
                    tooltip={this.__('widgetGeometryEdit')}
                    whenTouched={this.props.whenEditButtonTouched}
                />
            </gws.ui.Cell>}
        </gws.ui.Row>
    }
}



class Controller extends gws.Controller {
    cellView(props) {
    }

    activeCellView(props) {
    }

    formView(props) {
        return this.createElement(FormView, props)
    }
}



gws.registerTags({
    'ModelWidget.geometry': Controller,
})