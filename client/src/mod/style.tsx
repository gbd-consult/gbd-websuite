import * as React from 'react';

import * as gws from 'gws';
import * as style from 'gws/map/style';

import * as sidebar from './sidebar';
import * as storage from './storage';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Style';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as StyleController;


interface ViewProps extends gws.types.ViewProps {
    controller: StyleController;
    styleEditorActiveTab: number;
    styleEditorLabelEnabled: boolean;
    styleEditorCurrentSelector: string;
    styleEditorNewName: string;
    styleEditorValues: gws.api.StyleValues;
}


const StoreKeys = [
    'styleEditorActiveTab',
    'styleEditorLabelEnabled',
    'styleEditorCurrentSelector',
    'styleEditorNewName',
    'styleEditorValues',
];


const STORAGE_CATEGORY = 'styles';


class StyleForm extends gws.View<ViewProps> {
    render() {

        let bind = (prop, getter = null, setter = null) => {

            getter = getter || (x => x);
            setter = setter || (x => x);

            return {
                value: getter(this.props.styleEditorValues[prop]),
                whenChanged: val => cc.whenPropertyChanged(prop, setter(val))
            }
        }

        let cc = _master(this.props.controller);

        /*
                    <gws.ui.Group label={cc.__('modStyleName')} className="modStyleRenameControl">
                <gws.ui.TextInput
                    {...bind('styleEditorNewName')}/>
                <gws.ui.Button
                    tooltip={cc.__('modStyleRename')}
                    whenTouched={() => cc.renameStyle()}
                />
            </gws.ui.Group>

         */

        let labelPlacement = ['start', 'middle', 'end'].map(opt => <gws.ui.Toggle
                key={opt}
                className={'modStyleProp_label_placement_' + opt}
                tooltip={cc.__('modStyleProp_label_placement_' + opt)}
                {...bind('label_placement', val => val === opt, val => opt)}
            />,
        );

        let labelAlign = ['left', 'center', 'right'].map(opt => <gws.ui.Toggle
                key={opt}
                className={'modStyleProp_label_align_' + opt}
                tooltip={cc.__('modStyleProp_label_align_' + opt)}
                {...bind('label_align', val => val === opt, val => opt)}
            />,
        );

        let noLabel = false // this.props.stylerValues.with_label !== gws.api.StyleLabelOption.all;
        let noGeom = false // this.props.stylerValues.with_geometry !== gws.api.StyleGeometryOption.all;

        return <gws.ui.Tabs
            active={cc.getValue('styleEditorActiveTab')}
            whenChanged={n => cc.update({styleEditorActiveTab: n})}>

            <gws.ui.Tab label={cc.__('modStyleProp_with_geometry')}>

                <Form tabular>
                    <gws.ui.Group noBorder label={cc.__('modStyleEnabled')}>
                        <gws.ui.Toggle
                            type="checkbox"
                            {...bind('with_geometry', val => val === 'all', val => val ? 'all' : 'none')}
                        />
                    </gws.ui.Group>

                    <gws.ui.ColorPicker
                        disabled={noGeom}
                        label={cc.__('modStyleProp_fill')}
                        {...bind('fill')}
                    />
                    <gws.ui.ColorPicker
                        disabled={noGeom}
                        label={cc.__('modStyleProp_stroke')}
                        {...bind('stroke')}
                    />
                    <gws.ui.Slider
                        disabled={noGeom}
                        label={cc.__('modStyleProp_stroke_width')}
                        minValue={0} maxValue={20} step={1}
                        {...bind('stroke_width')}
                    />
                    <gws.ui.Slider
                        disabled={noGeom}
                        minValue={0} maxValue={20} step={1}
                        label={cc.__('modStyleProp_point_size')}
                        {...bind('point_size')}
                    />
                </Form>
            </gws.ui.Tab>

            <gws.ui.Tab label={cc.__('modStyleProp_with_label')}>
                <Form tabular>

                    <gws.ui.Group noBorder label={cc.__('modStyleEnabled')}>
                        <gws.ui.Toggle
                            type="checkbox"
                            {...bind('with_label', val => val === 'all', val => val ? 'all' : 'none')}
                        />
                    </gws.ui.Group>

                    <gws.ui.ColorPicker
                        disabled={noLabel}
                        label={cc.__('modStyleProp_fill')}
                        {...bind('label_fill')}
                    />
                    <gws.ui.ColorPicker
                        disabled={noLabel}
                        label={cc.__('modStyleProp_stroke')}
                        {...bind('label_stroke')}
                    />
                    <gws.ui.Slider
                        disabled={noLabel}
                        label={cc.__('modStyleProp_stroke_width')}
                        minValue={0} maxValue={20} step={1}
                        {...bind('label_stroke_width')}
                    />
                    <gws.ui.Slider
                        disabled={noLabel}
                        minValue={10} maxValue={40} step={1}
                        label={cc.__('modStyleProp_label_font_size')}
                        {...bind('label_font_size')}
                    />
                    <gws.ui.Group
                        noBorder
                        disabled={noLabel}
                        label={cc.__('modStyleProp_label_placement')}
                    >{labelPlacement}</gws.ui.Group>
                    <gws.ui.Group
                        noBorder
                        disabled={noLabel}
                        label={cc.__('modStyleProp_label_align')}
                    >{labelAlign}</gws.ui.Group>
                    <gws.ui.Group
                        noBorder
                        disabled={noLabel}
                        label={cc.__('modStyleProp_label_offset')}
                    >
                        <gws.ui.Slider
                            minValue={-100} maxValue={+100} step={1}
                            {...bind('label_offset_x')}
                        />
                        <gws.ui.Slider
                            minValue={-100} maxValue={+100} step={1}
                            {...bind('label_offset_y')}
                        />
                    </gws.ui.Group>
                </Form>
            </gws.ui.Tab>
        </gws.ui.Tabs>
    }
}

class StyleSidebarView extends gws.View<ViewProps> {
    render() {
        let styleNames = this.app.style.names.map(name => ({
            text: name,
            value: name,
        }));

        let cc = _master(this.props.controller);

        return <sidebar.Tab className="modStyleSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell flex>
                        <gws.ui.Title content={this.__('modStyleSidebarTitle')}/>
                    </Cell>
                    <Cell>
                        <gws.ui.Select items={styleNames} {...cc.bind('styleEditorCurrentSelector')}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <StyleForm {...this.props}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}

class StyleSidebar extends gws.Controller implements gws.types.ISidebarItem {

    iconClass = 'modStyleSidebarIcon';

    get tooltip() {
        return this.__('modStyleSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(StyleSidebarView, StoreKeys)
        );
    }
}

const UPDATE_DELAY = 200;


export class StyleController extends gws.Controller {
    uid = MASTER;

    updateTimer: any;


    async init() {
        this.update({
            styleEditorValues: {}
        });
        this.whenChanged('styleEditorCurrentSelector', () => this.loadStyle());
        // this.whenChanged('styleEditorValues', () => this.updateValues());

        let s = this.app.style.get('.modAnnotateFeature');
        this.update({
            styleEditorNewName: s.name,
            styleEditorValues: s.values,
        });
    }


    styleForm() {
        return this.createElement(
            this.connect(StyleForm, StoreKeys)
        );
    }


    loadStyle() {
        let name = this.getValue('styleEditorCurrentSelector');
        console.log('LOAD STYLE', name)
        let sty = this.app.style.at(name);
        let values = sty ? sty.values : {}

        this.update({
            styleEditorNewName: sty ? sty.name : '',
            styleEditorValues: {...style.DEFAULT_VALUES, ...values}
        });
    }

    whenPropertyChanged(prop, val) {
        let newValues = {
            ...this.getValue('styleEditorValues') || {},
            [prop]: val,
        };

        this.update({
            styleEditorValues: newValues
        });

        let name = this.getValue('styleEditorCurrentSelector');
        let sty = this.app.style.at(name);
        if (sty) {
            sty.update(newValues);
            clearTimeout(this.updateTimer);
            this.updateTimer = setTimeout(
                () => this.map.style.notifyChanged(this.map, sty.name),
                UPDATE_DELAY);
        }
    }


    // updateValues() {
    //     let name = this.getValue('styleEditorCurrentSelector');
    //     let sty = this.app.style.at(name);
    //     if (sty) {
    //         sty.update(this.getValue('styleEditorValues'));
    //         clearTimeout(this.updateTimer);
    //         this.updateTimer = setTimeout(() => this.map.style.notifyChanged(this.map, name), UPDATE_DELAY);
    //     }
    // }

    readStyles(data) {
        this.map.style.unserialize(data);
        this.map.style.notifyChanged(this.map);
        this.loadStyle();
    }

    writeStyles() {
        return this.map.style.serialize();
    }

    renameStyle() {


    }

}


export const tags = {

    [MASTER]: StyleController,
    'Sidebar.Style': StyleSidebar,
}