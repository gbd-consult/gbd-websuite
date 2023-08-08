import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.toggle.Props
}


class View extends gws.View<Props> {
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

class Controller extends gws.Controller {
    view(props) {
        return this.createElement(View, props)
    }
}


gws.registerTags({
    'ModelWidget.toggle': Controller,
})