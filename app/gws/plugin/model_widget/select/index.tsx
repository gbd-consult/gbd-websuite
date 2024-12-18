import * as React from 'react';

import * as gc from 'gc';
;

interface Props extends gc.types.ModelWidgetProps {
    widgetProps: gc.gws.plugin.model_widget.select.Props
}


class FormView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gc.ui.Select
            disabled={this.props.widgetProps.readOnly}
            value={value}
            items={this.props.widgetProps.items}
            withSearch={this.props.widgetProps.withSearch}
            whenChanged={this.props.whenChanged}
        />
    }
}

class CellView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        let text = ''

        for (let it of this.props.widgetProps.items)
            if (it.value === value)
                text = it.text

        return <gc.ui.TableCell content={text}/>;
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
    'ModelWidget.select': Controller,
})