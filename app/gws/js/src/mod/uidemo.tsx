import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: SidebarUIDemoController;
    uiDemoString: string;
    uiDemoNumber: number;
    uiDemoColor: string;
    uiDemoName: string;
    uiDemoDate: string;
    uiDemoAllDisabled: boolean;
    uiDemoBool: boolean;
    uiDemoFiles: FileList;
    uiDemoActiveTab: number,

    uiDemoMode: string;

    uiDemoUseTabs: boolean;
    uiDemoUseTitle: boolean;
    uiDemoUseClose: boolean;
    uiDemoUseFooter: boolean;
    uiDemoUseFrame: boolean;
    uiDemoUseRelative: boolean;

    uiDemoUseWidth: number;
    uiDemoUseHeight: number;

    uiDemoTableWidths: string,
    uiDemoTableHeaders: string,
    uiDemoTableFixedCols: number,
    uiDemoTableNumRows: number,

}

const StoreKeys = [
    'uiDemoString',
    'uiDemoNumber',
    'uiDemoColor',
    'uiDemoName',
    'uiDemoDate',
    'uiDemoAllDisabled',
    'uiDemoBool',
    'uiDemoFiles',
    'uiDemoActiveTab',

    'uiDemoMode',
    'uiDemoUseTabs',
    'uiDemoUseTitle',
    'uiDemoUseClose',
    'uiDemoUseFooter',
    'uiDemoUseFrame',
    'uiDemoUseRelative',

    'uiDemoUseWidth',
    'uiDemoUseHeight',

    'uiDemoTableWidths',
    'uiDemoTableHeaders',
    'uiDemoTableFixedCols',
    'uiDemoTableNumRows',

];

// https://en.wikipedia.org/wiki/List_of_most_common_surnames_in_Europe
const NAMES_1 = "Amato,Barbieri,Barone,Basile,Battaglia,Bellini,Benedetti,Bernardi,Bianchi,Bianco,Bruno,Caputo,Carbone,Caruso,Castelli,Cattaneo,Colombo,Conte,Conti,Coppola,Costa,D'Agostino,D'Amico,D'Angelo,De Angelis,De Luca,De Rosa,De Santis,Di Stefano,Esposito,Fabbri,Farina,Ferrara,Ferrari,Ferraro,Ferretti,Ferri,Fiore,Fontana,Franco,Galli,Gallo,Gatti,Gentile,Giordano,Giuliani,Grassi,Grasso,Greco,Guerra,Leone,Lombardi,Lombardo,Longo,Mancini,Marchetti,Mariani,Marini,Marino,Martinelli,Martini,Martino,Mazza,Messina,Montanari,Monti,Morelli,Moretti,Neri,Orlando,Pagano,Palmieri,Palumbo,Parisi,Pellegrini,Pellegrino,Piras,Poli,Ricci,Rinaldi,Riva,Rizzi,Rizzo,Romano,Romeo,Rossetti,Rossi,Ruggiero,Russo,Sala,Sanna,Santoro,Serra,Silvestri,Sorrentino,Testa,Valente,Valentini,Villa,Vitale"
    .split(',').sort();
const NAMES_2 = "Anderson,Brown,Campbell,Clark,MacDonald,Mitchell,Morrison,Murray,Paterson,Reid,Robertson,Ross,Scott,Smith,Stewart,Taylor,Thomson,Watson,Wilson,Young"
    .split(',').sort();


const NAMES = NAMES_1.map(c => ({text: c, value: 'name:' + c}));
const NAMES_GROUPED = (() => {
        let items = [], last = '';
        for (let n of NAMES_1) {
            if (last[0] !== n[0]) {
                let m = NAMES_2.shift();
                items.push({text: m, value: 'name:' + m, level: 1});
            }
            items.push({text: n, value: 'name:' + n, level: 2});
            last = n
        }
        return items;
    }
)();

class TableForm extends gws.View<ViewProps> {
    render() {
        let bind = name => value => this.props.controller.update({[name]: value});

        let getRow = r => [
            String(r),
            <gws.ui.Select
                value={this.props.uiDemoName}
                label="select"
                items={NAMES.slice(0, 5)}
                whenChanged={bind('uiDemoName')}
            />,
            <gws.ui.TextInput
                value={this.props.uiDemoString}
                label="text input"
                whenChanged={bind('uiDemoString')}
            />,
            <gws.ui.TextInput
                value={this.props.uiDemoString}
                label="text input"
                whenChanged={bind('uiDemoString')}
            />,
            <gws.ui.Group label="options">
                <gws.ui.Toggle
                    inline
                    type="checkbox"
                    value={this.props.uiDemoBool}
                    whenChanged={bind('uiDemoBool')}
                />
            </gws.ui.Group>,
            <gws.ui.DateInput
                value={this.props.uiDemoDate}
                label="date"
                withClear
                whenChanged={bind('uiDemoDate')}
            />,
        ];

        let tp = {getRow}, s;

        let split = s => s.trim().split(' ').map(p => p.trim());

        if (s = this.props.uiDemoTableWidths) {
            tp['widths'] = split(s).map(Number);
        }

        if (s = this.props.uiDemoTableFixedCols) {
            tp['fixedColumns'] = s
        }

        if (s = this.props.uiDemoTableHeaders) {
            tp['headers'] = split(s)
        }

        if (s = this.props.uiDemoTableNumRows) {
            tp['numRows'] = s;
        }

        return <div style={{width: '100%', height: '100%', display: 'flex'}}>
            <div style={{paddingRight: 50, width: 200}}>
                <gws.ui.TextInput
                    value={this.props.uiDemoTableWidths}
                    label="widths"
                    whenChanged={bind('uiDemoTableWidths')}
                />
                <gws.ui.TextInput
                    value={this.props.uiDemoTableHeaders}
                    label="top"
                    whenChanged={bind('uiDemoTableHeaders')}
                />
                <gws.ui.NumberInput
                    step={1}
                    value={this.props.uiDemoTableFixedCols}
                    label="left"
                    whenChanged={bind('uiDemoTableFixedCols')}
                />
                <gws.ui.NumberInput
                    step={1}
                    value={this.props.uiDemoTableNumRows}
                    label="num rows"
                    whenChanged={bind('uiDemoTableNumRows')}
                />
            </div>
            <div style={{height: '100%', flex: 1, overflow: 'auto' as any}}>
                <gws.ui.Table {...tp}/>
            </div>
        </div>
    }
}


class TabularForm extends gws.View<ViewProps> {
    render() {
        let bind = name => value => this.props.controller.update({[name]: value})

        return <Form tabular>
            <gws.ui.Select
                value={this.props.uiDemoName}
                label="select"
                items={NAMES.slice(0, 5)}
                whenChanged={bind('uiDemoName')}
            />
            <gws.ui.TextInput
                value={this.props.uiDemoString}
                label="text input"
                whenChanged={bind('uiDemoString')}
            />
            <gws.ui.DateInput
                value={this.props.uiDemoDate}
                label="date"
                minValue={"2020-11-05"}
                maxValue={"2020-12-13"}
                withClear
                whenChanged={bind('uiDemoDate')}
            />
            <gws.ui.Toggle
                type="checkbox"
                label="checkbox"
                value={this.props.uiDemoBool}
                whenChanged={bind('uiDemoBool')}
            />
            <gws.ui.Toggle
                type="radio"
                label="radio"
                value={this.props.uiDemoBool}
                whenChanged={bind('uiDemoBool')}
            />
            <gws.ui.ColorPicker
                value={this.props.uiDemoColor}
                label="Hintergrundfarbe"
                whenChanged={bind('uiDemoColor')}
            />
            <gws.ui.Group label="options" vertical>
                <gws.ui.Toggle
                    inline
                    type="checkbox"
                    label="checkbox"
                    value={this.props.uiDemoBool}
                    whenChanged={bind('uiDemoBool')}
                />
                <gws.ui.Toggle
                    inline
                    type="radio"
                    label="radio"
                    value={this.props.uiDemoBool}
                    whenChanged={bind('uiDemoBool')}
                />
            </gws.ui.Group>
        </Form>
    }
}

class BigForm extends gws.View<ViewProps> {
    render() {
        let bind = name => value => this.props.controller.update({[name]: value})

        return <Form>
            <Row top>
                <Cell>
                    <gws.ui.List
                        value={this.props.uiDemoName}
                        label="list"
                        items={NAMES}
                        whenChanged={bind('uiDemoName')}
                    />
                </Cell>
                <Cell>
                    <gws.ui.List
                        value={this.props.uiDemoName}
                        label="list/buttons"
                        items={NAMES}
                        rightButton={it => <gws.ui.Button className="uiClearButton"/>}
                        whenChanged={bind('uiDemoName')}
                    />
                </Cell>
                <Cell flex>
                    <Form>
                        <Row>
                            <Cell flex>
                                <gws.ui.Select
                                    value={this.props.uiDemoName}
                                    label="select"
                                    items={NAMES.filter(x => x.text.match(/^Mar/))}
                                    whenChanged={bind('uiDemoName')}
                                />
                            </Cell>
                            <Cell flex>
                                <gws.ui.Select
                                    value={this.props.uiDemoName}
                                    label="disabled"
                                    disabled
                                    items={NAMES}
                                    whenChanged={bind('uiDemoName')}
                                />
                            </Cell>
                        </Row>
                        <Row>
                            <Cell flex>
                                <gws.ui.Select
                                    value={this.props.uiDemoName}
                                    label="search"
                                    items={NAMES}
                                    withSearch
                                    whenChanged={bind('uiDemoName')}
                                />
                            </Cell>
                            <Cell flex>
                                <gws.ui.Select
                                    value={this.props.uiDemoName}
                                    label="search+clear"
                                    items={NAMES}
                                    withClear
                                    withSearch
                                    whenChanged={bind('uiDemoName')}
                                />
                            </Cell>
                        </Row>
                        <Row>
                            <Cell flex>
                                <gws.ui.Select
                                    value={this.props.uiDemoName}
                                    label="combo"
                                    withCombo
                                    items={NAMES}
                                    whenChanged={bind('uiDemoName')}
                                />
                            </Cell>
                            <Cell flex>
                                <gws.ui.Select
                                    value={this.props.uiDemoName}
                                    label="levels"
                                    withSearch
                                    items={NAMES_GROUPED}
                                    whenChanged={bind('uiDemoName')}
                                />
                            </Cell>
                        </Row>
                    </Form>
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.DateInput
                        value={this.props.uiDemoDate}
                        label="date"
                        minValue={"2020-11-05"}
                        maxValue={"2020-12-13"}
                        withClear
                        locale={this.props.controller.app.locale}
                        whenChanged={bind('uiDemoDate')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.ColorPicker
                        value={this.props.uiDemoColor}
                        label="color"
                        whenChanged={bind('uiDemoColor')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.FileInput
                        label="file"
                        multiple
                        value={this.props.uiDemoFiles}
                        whenChanged={bind('uiDemoFiles')}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        value={this.props.uiDemoString}
                        label="textInput"
                        whenChanged={bind('uiDemoString')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        value={this.props.uiDemoString}
                        label="textInput+clear"
                        withClear
                        whenChanged={bind('uiDemoString')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        disabled
                        value={this.props.uiDemoString}
                        label="disabled"
                        whenChanged={bind('uiDemoString')}
                    />
                </Cell>
            </Row>

            <Row>
                <Cell>
                    <gws.ui.NumberInput
                        value={this.props.uiDemoNumber}
                        label="float"
                        locale={this.props.controller.app.locale}
                        whenChanged={bind('uiDemoNumber')}
                    />
                </Cell>
                <Cell>
                    <gws.ui.NumberInput
                        value={this.props.uiDemoNumber}
                        label="int, step 5"
                        minValue={-200}
                        maxValue={200}
                        step={5}
                        withClear
                        whenChanged={bind('uiDemoNumber')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.Slider
                        value={this.props.uiDemoNumber}
                        label="no step"
                        minValue={-200}
                        maxValue={200}
                        whenChanged={bind('uiDemoNumber')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.Slider
                        value={this.props.uiDemoNumber}
                        label="step 50"
                        minValue={-200}
                        maxValue={200}
                        step={50}
                        whenChanged={bind('uiDemoNumber')}
                    />
                </Cell>
            </Row>

            <Row top>
                <Cell flex>
                    <gws.ui.TextArea
                        value={this.props.uiDemoString}
                        label="area"
                        whenChanged={bind('uiDemoString')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.Progress
                        value={100 * (this.props.uiDemoNumber + 200) / 400}
                        label="progress"
                    />
                </Cell>
            </Row>

            <Row top>
                <Cell>
                    <gws.ui.Toggle
                        type="checkbox"
                        label="checkbox"
                        value={this.props.uiDemoBool}
                        whenChanged={bind('uiDemoBool')}
                    />
                </Cell>

                <Cell>
                    <gws.ui.Group label="options">
                        <gws.ui.Toggle
                            inline
                            type="checkbox"
                            label="checkbox"
                            value={this.props.uiDemoBool}
                            whenChanged={bind('uiDemoBool')}
                        />
                        <gws.ui.Toggle
                            inline
                            type="radio"
                            label="radio"
                            value={this.props.uiDemoBool}
                            whenChanged={bind('uiDemoBool')}
                        />
                        <gws.ui.Toggle
                            inline
                            type="radio"
                            label="disabled"
                            disabled
                            value={this.props.uiDemoBool}
                            whenChanged={bind('uiDemoBool')}
                        />
                    </gws.ui.Group>
                </Cell>
                <Cell>
                    <gws.ui.Group label="options" vertical>
                        <gws.ui.Toggle
                            inline
                            type="checkbox"
                            label="checkbox"
                            value={this.props.uiDemoBool}
                            whenChanged={bind('uiDemoBool')}
                        />
                        <gws.ui.Toggle
                            inline
                            type="radio"
                            label="radio"
                            value={this.props.uiDemoBool}
                            whenChanged={bind('uiDemoBool')}
                        />
                    </gws.ui.Group>
                </Cell>
            </Row>

        </Form>
    }
}

class ButtonsDemo extends gws.View<ViewProps> {
    render() {
        return <Form>
            <Row>
                <Cell>
                    <gws.ui.Button label="text"/>
                </Cell>
                <Cell>
                    <gws.ui.Button label="primary" primary/>
                </Cell>
                <Cell>
                    <gws.ui.Button label="disabled" disabled/>
                </Cell>
            </Row>
            <Row>
                <Cell>
                    <gws.ui.Button className="uiDemoIcon"/>
                </Cell>
                <Cell>
                    <gws.ui.Button className="uiDemoIcon" badge="15"/>
                </Cell>
                <Cell>
                    <gws.ui.Button className="uiDemoIcon" disabled/>
                </Cell>
            </Row>
        </Form>


    }


}

class DialogContent extends gws.View<ViewProps> {
    render() {
        let bind = name => value => this.props.controller.update({[name]: value})

        return <gws.ui.Tabs
            active={this.props.uiDemoActiveTab}
            whenChanged={bind('uiDemoActiveTab')}
        >
            <gws.ui.Tab label="table">
                <TableForm {...this.props}/>
            </gws.ui.Tab>
            <gws.ui.Tab label="form">
                <BigForm {...this.props}/>
            </gws.ui.Tab>
            <gws.ui.Tab label="tabular">
                <TabularForm {...this.props}/>
            </gws.ui.Tab>
            <gws.ui.Tab label="buttons">
                <ButtonsDemo {...this.props}/>
            </gws.ui.Tab>
            <gws.ui.Tab label="text">
                <p>William Shakespeare was an English poet, playwright, and actor, widely regarded as the greatest
                    writer in the English language and the world's greatest dramatist</p>
            </gws.ui.Tab>
            <gws.ui.Tab label="Unknown" disabled>
            </gws.ui.Tab>
        </gws.ui.Tabs>
    }
}


class SidebarBody extends gws.View <ViewProps> {

    render() {

        let
            bind = name => value => this.props.controller.update({[name]: value}),
            set = (name, value) => () => this.props.controller.update({[name]: value});

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content='UI Demo'/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Form>
                    <Row>
                        <Cell>
                            <gws.ui.NumberInput
                                label="width"
                                minValue={20} step={1}
                                value={this.props.uiDemoUseWidth}
                                whenChanged={bind('uiDemoUseWidth')}
                            />
                        </Cell>
                        <Cell>
                            <gws.ui.NumberInput
                                label="height"
                                minValue={20} step={1}
                                value={this.props.uiDemoUseHeight}
                                whenChanged={bind('uiDemoUseHeight')}
                            />
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex>
                            <gws.ui.Group vertical>
                                <gws.ui.Toggle
                                    inline
                                    type="checkbox"
                                    label="relative"
                                    value={this.props.uiDemoUseRelative}
                                    whenChanged={bind('uiDemoUseRelative')}
                                />
                                <gws.ui.Toggle
                                    inline
                                    type="checkbox"
                                    label="tabs"
                                    value={this.props.uiDemoUseTabs}
                                    whenChanged={bind('uiDemoUseTabs')}
                                />
                                <gws.ui.Toggle
                                    inline
                                    type="checkbox"
                                    label="title"
                                    value={this.props.uiDemoUseTitle}
                                    whenChanged={bind('uiDemoUseTitle')}
                                />
                                <gws.ui.Toggle
                                    inline
                                    type="checkbox"
                                    label="close"
                                    value={this.props.uiDemoUseClose}
                                    whenChanged={bind('uiDemoUseClose')}
                                />
                                <gws.ui.Toggle
                                    inline
                                    type="checkbox"
                                    label="footer"
                                    value={this.props.uiDemoUseFooter}
                                    whenChanged={bind('uiDemoUseFooter')}
                                />
                                <gws.ui.Toggle
                                    inline
                                    type="checkbox"
                                    label="frame"
                                    value={this.props.uiDemoUseFrame}
                                    whenChanged={bind('uiDemoUseFrame')}
                                />
                            </gws.ui.Group>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoMode', 'dialog')}
                                label="dialog"/>
                        </Cell>
                    </Row>
                    <Row>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoMode', 'alertError')}
                                label="err"/>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoMode', 'alertInfo')}
                                label="info"/>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoMode', 'popup')}
                                label="popup"/>
                        </Cell>
                    </Row>

                </Form>

                <pre style={{padding: 10}}>
                    uiDemoString: {this.props.uiDemoString}<br/>
                    uiDemoNumber: {this.props.uiDemoNumber}<br/>
                    uiDemoColor: {this.props.uiDemoColor}<br/>
                    uiDemoName: {this.props.uiDemoName}<br/>
                    uiDemoDate: {this.props.uiDemoDate}<br/>
                </pre>

            </sidebar.TabBody>

            <sidebar.TabFooter>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class OverlayView extends gws.View<ViewProps> {
    render() {
        let dm = this.props.uiDemoMode;

        if (!dm)
            return null;

        let close = () => this.props.controller.update({
            uiDemoMode: ''
        });

        let buttons = [
            <gws.ui.Button
                whenTouched={close}
                primary
                label="ok"/>,
            <gws.ui.Button
                whenTouched={close}
                label="cancel"/>
        ];

        let CENTER_BOX = (w, h) => ({
            width: w,
            height: h,
            marginLeft: -(w >> 1),
            marginTop: -(h >> 1),
        });

        let dlgOpts = {
            title: this.props.uiDemoUseTitle ? "Dialog Title" : null,
            whenClosed: this.props.uiDemoUseClose ? close : null,
            buttons: this.props.uiDemoUseFooter ? buttons : null,
            frame: this.props.uiDemoUseFrame ? '/chess.png' : null,
        };

        let DIALOG_PADDING = 35;

        if (this.props.uiDemoUseRelative) {
            dlgOpts['style'] = {
                left: DIALOG_PADDING,
                top: DIALOG_PADDING,
                right: DIALOG_PADDING,
                bottom: DIALOG_PADDING,
                width: 'auto',
                height: 'auto',
                margin: 'auto',
            }
        } else {
            dlgOpts['style'] = CENTER_BOX(this.props.uiDemoUseWidth, this.props.uiDemoUseHeight)
        }

        if (dm === 'dialog') {
            if (this.props.uiDemoUseTabs) {
                return <gws.ui.Dialog {...dlgOpts}><DialogContent {...this.props}/></gws.ui.Dialog>
            } else {
                return <gws.ui.Dialog {...dlgOpts}><TableForm {...this.props}/></gws.ui.Dialog>
            }
        }

        if (dm === 'alertError') {
            return <gws.ui.Alert
                title="Error"
                whenClosed={close}
                error="Error message"
                details="Some details about the error message. Lorem ipsum dolor sit amet, consectetur adipiscing elit"
            />
        }

        if (dm === 'alertInfo') {
            return <gws.ui.Alert
                title="Info"
                whenClosed={close}
                info="Info message"
                details="Some details about the info message"
            />
        }

        if (dm === 'popup') {
            return <gws.ui.Popup
                style={{
                    left: 100,
                    top: 100,
                    width: 200,
                    height: 200,
                    background: 'white'
                }}
                whenClosed={close}
            >
                <gws.ui.Text content="POPUP CONTENT"/>
            </gws.ui.Popup>
        }

    }
}


class SidebarUIDemoController extends gws.Controller implements gws.types.ISidebarItem {
    tooltip = '';
    iconClass = '';


    async init() {
        this.update({
            uiDemoString: 'string',
            uiDemoNumber: 13,
            uiDemoColor: 'rgba(255,200,10,0.9)',
            uiDemoName: 'name:Marino',
            uiDemoDate: '2020-11-22',

            uiDemoMode: 'dialog',
            uiDemoActiveTab: 1,

            uiDemoUseTabs: true,
            uiDemoUseTitle: true,
            uiDemoUseClose: false,
            uiDemoUseFooter: true,
            uiDemoUseFrame: false,
            uiDemoUseRelative: true,

            uiDemoUseWidth: 800,
            uiDemoUseHeight: 580,

            uiDemoTableWidths: '40 80 150 100 260 0',
            uiDemoTableHeaders: 'A B C D E F',
            uiDemoTableFixedCols: 2,
            uiDemoTableNumRows: 20,


        });
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(OverlayView, StoreKeys)
        );
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarBody, StoreKeys),
        );
    }
}

export const tags = {
    'Shared.UIDemo': SidebarUIDemoController,
    'Sidebar.UIDemo': SidebarUIDemoController
};
