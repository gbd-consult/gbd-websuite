import * as React from 'react';

import * as gc from 'gc';
;

interface Props extends gc.types.ModelWidgetProps {
    widgetProps: gc.gws.plugin.model_widget.geometry.Props;
    whenNewButtonTouched?: () => void;
    whenEditButtonTouched?: () => void;
    whenEditTextButtonTouched?: () => void;
}

class FormView extends gc.View<Props> {
    render() {
        let cc = this.props.controller;
        let field = this.props.field;
        let hasGeom = Boolean(this.props.feature.geometry);
        let isDrawing = cc.app.activeTool.tag.includes('Draw');
        let isInline = this.props.widgetProps.isInline;
        let withText = this.props.widgetProps.withText;

        return <gc.ui.Row>
            {!hasGeom && this.props.whenNewButtonTouched && <gc.ui.Cell>
                <gc.ui.Button
                    {...gc.lib.cls('cmpFormDrawGeometryButton', isDrawing && 'isActive', isInline && 'isInline')}
                    tooltip={this.__('widgetGeometryNew')}
                    whenTouched={this.props.whenNewButtonTouched}
                />
            </gc.ui.Cell>}
            {hasGeom && this.props.whenEditButtonTouched && <gc.ui.Cell>
                <gc.ui.Button
                    {...gc.lib.cls('cmpFormEditGeometryButton', isInline && 'isInline')}
                    tooltip={this.__('widgetGeometryEdit')}
                    whenTouched={this.props.whenEditButtonTouched}
                />
            </gc.ui.Cell>}
            {withText && this.props.whenEditTextButtonTouched && <gc.ui.Cell spaced>
                <gc.ui.Button
                    {...gc.lib.cls('cmpFormGeometryTextButton', isInline && 'isInline')}
                    tooltip={this.__('widgetGeometryEditText')}
                    whenTouched={this.props.whenEditTextButtonTouched}
                />
            </gc.ui.Cell>}
        </gc.ui.Row>
    }
}



class Controller extends gc.Controller {
    cellView(props) {
    }

    activeCellView(props) {
    }

    formView(props) {
        return this.createElement(FormView, props)
    }
}



gc.registerTags({
    'ModelWidget.geometry': Controller,
})