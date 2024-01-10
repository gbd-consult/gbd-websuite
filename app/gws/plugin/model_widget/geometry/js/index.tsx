import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.geometry.Props;
    whenNewButtonTouched?: () => void;
    whenEditButtonTouched?: () => void;
    whenEditTextButtonTouched?: () => void;
}

class FormView extends gws.View<Props> {
    render() {
        let cc = this.props.controller;
        let field = this.props.field;
        let hasGeom = Boolean(this.props.feature.geometry);
        let isDrawing = cc.app.activeTool.tag.includes('Draw');
        let isInline = this.props.widgetProps.isInline;
        let withText = this.props.widgetProps.withText;

        return <gws.ui.Row>
            {!hasGeom && this.props.whenNewButtonTouched && <gws.ui.Cell>
                <gws.ui.Button
                    {...gws.lib.cls('cmpFormDrawGeometryButton', isDrawing && 'isActive', isInline && 'isInline')}
                    tooltip={this.__('widgetGeometryNew')}
                    whenTouched={this.props.whenNewButtonTouched}
                />
            </gws.ui.Cell>}
            {hasGeom && this.props.whenEditButtonTouched && <gws.ui.Cell>
                <gws.ui.Button
                    {...gws.lib.cls('cmpFormEditGeometryButton', isInline && 'isInline')}
                    tooltip={this.__('widgetGeometryEdit')}
                    whenTouched={this.props.whenEditButtonTouched}
                />
            </gws.ui.Cell>}
            {withText && this.props.whenEditTextButtonTouched && <gws.ui.Cell spaced>
                <gws.ui.Button
                    {...gws.lib.cls('cmpFormGeometryTextButton', isInline && 'isInline')}
                    tooltip={this.__('widgetGeometryEditText')}
                    whenTouched={this.props.whenEditTextButtonTouched}
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