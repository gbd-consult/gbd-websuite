import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.textarea.Props
}


class FormView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.TextArea
            height={this.props.widgetProps.height}
            disabled={this.props.widgetProps.readOnly}
            placeholder={this.props.widgetProps.placeholder || ''}
            value={value}
            whenChanged={this.props.whenChanged}
        />
    }
}

class CellView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name] || '';
        return <gws.ui.TableCell content={value}/>;
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
    'ModelWidget.textarea': Controller,
})