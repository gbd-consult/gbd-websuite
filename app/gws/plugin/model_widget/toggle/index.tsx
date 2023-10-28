import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.toggle.Props
}


class FormView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.Toggle
            disabled={this.props.widgetProps.readOnly}
            value={gws.lib.isEmpty(value) ? null : Boolean(value)}
            whenChanged={this.props.whenChanged}
            type={this.props.widgetProps.kind}
        />
    }
}

class CellView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        if (value) {
            return <gws.ui.TableCell content="✔" />
        }

        return null
    }
}

class Controller extends gws.Controller {
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





gws.registerTags({
    'ModelWidget.toggle': Controller,
})