import * as React from 'react';

import * as gws from 'gws';
import * as components from 'gws/components';

const {Form, Row, Cell} = gws.ui.Layout;

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.file.Props
}

// see app/gws/plugin/model_field/file/__init__.py
export interface FileOutputProps {
    downloadUrl?: string
    extension?: string
    label?: string
    previewUrl?: string
    size?: number
}


class FormView extends gws.View<gws.types.ModelWidgetProps> {
    download(fp: FileOutputProps) {
        let u = fp.downloadUrl || '';
        gws.lib.downloadUrl(u, u.split('/').pop())
    }


    render() {
        let field = this.props.field;
        let cc = this.props.controller;
        let value = this.props.values[field.name];
        let fp = this.props.feature.getAttribute(field.name) as FileOutputProps;

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
                    <gws.ui.FileInput
                        // accept={editor.accept}
                        value={value}
                        whenChanged={this.props.whenChanged}
                        tooltip={this.__('widgetFileUpload')}
                    />
                </Cell>
                <Cell flex/>
                {fp && <Cell>
                    <gws.ui.Button
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

class CellView extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let cc = this.props.controller;
        let value = this.props.values[field.name];
        let fp = this.props.feature.getAttribute(field.name) as FileOutputProps;

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
    'ModelWidget.file': Controller,
})
