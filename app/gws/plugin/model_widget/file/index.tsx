import * as React from 'react';

import * as gc from 'gc';
;
import * as components from 'gc/components';

const {Form, Row, Cell} = gc.ui.Layout;

interface Props extends gc.types.ModelWidgetProps {
    widgetProps: gc.gws.plugin.model_widget.file.Props
}



class FormView extends gc.View<gc.types.ModelWidgetProps> {
    download(fp: gc.types.ServerFileProps) {
        let u = fp.downloadUrl || '';
        gc.lib.downloadUrl(u, u.split('/').pop())
    }


    render() {
        let field = this.props.field;
        let cc = this.props.controller;
        let value = this.props.values[field.name];
        let fp = this.props.feature.getAttribute(field.name) as gc.types.ServerFileProps;

        return <div className="cmpFormList">
            {fp && <components.file.File
                controller={cc}
                item={{
                    label: fp.label,
                    extension: fp.extension,
                    previewUrl: fp.previewUrl,
                }}
            />}
            <Row className="cmpFormListToolbar">
                <Cell>
                    <gc.ui.FileInput
                        // accept={editor.accept}
                        value={value}
                        whenChanged={this.props.whenChanged}
                        tooltip={this.__('widgetFileUpload')}
                    />
                </Cell>
                <Cell flex/>
                {fp && <Cell>
                    <gc.ui.Button
                        className='cmpFormFileDownloadButton'
                        disabled={!fp.downloadUrl}
                        whenTouched={() => this.download(fp)}
                        tooltip={this.__('widgetFileDownload')}
                    />
                </Cell>}
            </Row>
        </div>
    }
}

class CellView extends gc.View<gc.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let cc = this.props.controller;
        let value = this.props.values[field.name];
        let fp = this.props.feature.getAttribute(field.name) as gc.types.ServerFileProps;

        return <div className="cmpFormList">
            {fp && <components.file.File
                controller={cc}
                item={{
                    label: fp.label,
                    extension: fp.extension,
                    previewUrl: fp.previewUrl,
                }}
            />}
        </div>
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
    'ModelWidget.file': Controller,
})
