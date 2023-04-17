import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.textarea.Props
}


class View extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.TextArea
            height={this.props.widgetProps.height}
            disabled={this.props.widgetProps.readOnly}
            value={value}
            whenChanged={this.props.whenChanged}
        />
    }
}

class Controller extends gws.Controller {
    view(props) {
        return this.createElement(View, props)
    }
}


gws.registerTags({
    'ModelWidget.textarea': Controller,
})