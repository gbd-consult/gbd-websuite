import * as React from 'react';

import * as gc from 'gc';
;

interface Props extends gc.types.ModelWidgetProps {
    widgetProps: gc.gws.plugin.model_widget.toggle.Props
}


class FormView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gc.ui.Toggle
            disabled={this.props.widgetProps.readOnly}
            value={gc.lib.isEmpty(value) ? null : Boolean(value)}
            whenChanged={this.props.whenChanged}
            type={this.props.widgetProps.kind}
        />
    }
}

class CellView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        if (value) {
            return <gc.ui.TableCell content="✔" />
        }

        return null
    }
}

class Controller extends gc.Controller {
    cellView(props) {
        return this.createElement(CellView, props)
    }

    activeCellView(props) {
        return this.createElement(FormView, props)
    }

    formView(props) {
        return this.createElement(FormView, props)
    }
}





gc.registerTags({
    'ModelWidget.toggle': Controller,
})