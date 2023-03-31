import * as React from 'react';

import * as gws from 'gws';

class View extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.NumberInput
            step={this.props.options.step || 1}
            locale={this.app.locale}
            value={gws.lib.isEmpty(value) ? null : Number(value)}
            whenChanged={v => this.props.when('changed', this.props.controller, field, v)}
            whenEntered={v => this.props.when('entered', this.props.controller, field, v)}
            disabled={this.props.readOnly}
        />
    }
}

gws.registerTags({
    'ModelWidget.integer': View,
})