import * as React from 'react';

import * as gc from 'gc';

class FormView extends gc.View<gc.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        return <gc.ui.DateInput
            disabled={this.props.widgetProps.readOnly}
            value={gc.lib.isEmpty(value) ? '' : String(value)}
            locale={this.app.locale}
            whenChanged={this.props.whenChanged}
        />
    }
}

class CellView extends gc.View<gc.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        let dmy = gc.ui.util.iso2dmy(value);
        let lo = this.app.locale;
        let v = dmy ? gc.ui.util.formatDate(dmy, lo.dateFormatShort, lo) : ''

        return <gc.ui.TableCell align="right" content={v}/>
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
    'ModelWidget.date': Controller,
})