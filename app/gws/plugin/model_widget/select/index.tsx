import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.select.Props
}


class FormView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.Select
            disabled={this.props.widgetProps.readOnly}
            value={value}
            items={this.props.widgetProps.items}
            withSearch={this.props.widgetProps.withSearch}
            whenChanged={this.props.whenChanged}
        />
    }
}

class CellView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        let text = ''

        for (let it of this.props.widgetProps.items)
            if (it.value === value)
                text = it.text

        return <gws.ui.TableCell content={text}/>;
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
    'ModelWidget.select': Controller,
})