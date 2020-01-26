import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';
import * as storage from './common/storage';

let {Form, Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: StyleController;
    styleEditorLabelEnabled: boolean;
    styleEditorName: string;
    styleEditorNewName: string;
    styleEditorValues: gws.api.StyleValues;
}


const StoreKeys = [
    'styleEditorLabelEnabled',
    'styleEditorName',
    'styleEditorNewName',
    'styleEditorValues',
];


const STORAGE_CATEGORY = 'styles';


class StyleForm extends gws.View<ViewProps> {
    render() {

        let cc = this.props.controller;
        let sv = this.props.styleEditorValues;

        return <Form tabular>
            <gws.ui.Group label={cc.__('modStyleName')} className="modStyleRenameControl">
                <gws.ui.TextInput
                    {...cc.bind('styleEditorNewName')}/>
                <gws.ui.Button
                    tooltip={cc.__('modStyleRename')}
                    whenTouched={() => cc.renameStyle()}
                />
            </gws.ui.Group>
            <gws.ui.ColorPicker
                label={cc.__('modStyleProp_fill')}
                {...cc.bind('styleEditorValues.fill')}/>
            <gws.ui.ColorPicker
                label={cc.__('modStyleProp_stroke')}
                {...cc.bind('styleEditorValues.stroke')}/>
            <gws.ui.Slider
                label={cc.__('modStyleProp_stroke_width')}
                minValue={0}
                maxValue={20}
                step={1}
                {...cc.bind('styleEditorValues.stroke_width')}/>
            <gws.ui.Slider
                minValue={0}
                maxValue={20}
                step={1}
                label={cc.__('modStyleProp_point_size')}
                {...cc.bind('styleEditorValues.point_size')}/>
            <gws.ui.ColorPicker
                label={cc.__('modStyleProp_label_background')}
                {...cc.bind('styleEditorValues.label_background')}/>
            <gws.ui.ColorPicker
                label={cc.__('modStyleProp_label_fill')}
                {...cc.bind('styleEditorValues.label_fill')}/>
            <gws.ui.Slider
                minValue={6}
                maxValue={20}
                step={1}
                label={cc.__('modStyleProp_label_font_size')}
                {...cc.bind('styleEditorValues.label_font_size')}/>
            <gws.ui.Slider
                minValue={-100}
                maxValue={+100}
                step={1}
                label={cc.__('modStyleProp_label_offset_x')}
                {...cc.bind('styleEditorValues.label_offset_x')}/>
            <gws.ui.Slider
                minValue={-100}
                maxValue={+100}
                step={1}
                label={cc.__('modStyleProp_label_offset_y')}
                {...cc.bind('styleEditorValues.label_offset_y')}/>
        </Form>
    }
}

class SidebarBody extends gws.View<ViewProps> {

    render() {

        let cc = this.props.controller;

        let styleNames = this.app.style.names.sort().map(name => ({
            text: name,
            value: name,
        }));

        return <sidebar.Tab className="modStyleSidebar">

            <sidebar.TabHeader>
                <Row>
                    <Cell flex>
                        <gws.ui.Title content={this.__('modStyleSidebarTitle')}/>
                    </Cell>
                    <Cell>
                        <gws.ui.Select items={styleNames} {...cc.bind('styleEditorName')}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <StyleForm {...this.props}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <storage.ReadAuxButton
                        controller={cc}
                        category={STORAGE_CATEGORY}
                        whenDone={data => cc.readStyles(data)}
                    />
                    {<storage.WriteAuxButton
                        controller={this.props.controller}
                        category={STORAGE_CATEGORY}
                        data={cc.writeStyles()}
                    />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>
    }
}

const UPDATE_DELAY = 500;

class StyleController extends gws.Controller implements gws.types.ISidebarItem {

    updateTimer: any;

    iconClass = 'modStyleSidebarIcon';

    async init() {
        this.update({
            styleEditorValues: {}
        });
        this.whenChanged('styleEditorName', () => this.loadStyle());
        this.whenChanged('styleEditorValues', () => this.updateValues());
    }

    get tooltip() {
        return this.__('modStyleSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarBody, StoreKeys)
        );
    }



    loadStyle() {
        let name = this.getValue('styleEditorName');
        let s = this.app.style.get(name);
        this.update({
            styleEditorNewName: s.name,
            styleEditorValues: s.values,
        });
    }

    updateValues() {
        let name = this.getValue('styleEditorName');
        this.app.style.update(name, this.getValue('styleEditorValues'));
        clearTimeout(this.updateTimer);
        this.updateTimer = setTimeout(() => this.map.style.notifyChange(this.map, name), UPDATE_DELAY);
    }

    readStyles(data) {
        this.map.style.unserialize(data);
        this.map.style.notifyChange(this.map);
        this.loadStyle();
    }

    writeStyles() {
        return this.map.style.serialize();
    }

    renameStyle() {


    }

}


export const tags = {
    'Sidebar.Style': StyleController,
};
