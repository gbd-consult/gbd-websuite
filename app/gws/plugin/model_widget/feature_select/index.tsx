import * as React from 'react';

import * as gc from 'gc';
;

interface Props extends gc.types.ModelWidgetProps {
    features: Array<gc.types.IFeature>;
    widgetProps: gc.gws.plugin.model_widget.feature_select.Props
}

class FormView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let feature = this.props.values[field.name];
        let items = [];
        let fmap = {};

        for (let f of this.props.features) {
            fmap[f.uid] = f;
            items.push({text: f.views.title, value: f.uid})
        }

        return <gc.ui.Select
            value={feature ? feature.uid : null}
            items={items}
            withSearch={this.props.widgetProps.withSearch}
            whenChanged={v => this.props.whenChanged(fmap[v])}
        />;
    }
}

class CellView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let feature = this.props.values[field.name];
        let text = feature ? feature.views.title : '';
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
    'ModelWidget.featureSelect': Controller,
})
