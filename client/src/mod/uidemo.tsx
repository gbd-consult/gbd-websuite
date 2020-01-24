import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: SidebarUIDemoController;
    uiDemoString: string;
    uiDemoNumber: number;
    uiDemoColor: string;
    uiDemoCountry: string;
    uiDemoDate: string;
    uiDemoAllDisabled: boolean;
    uiDemoBool: boolean;
    uiDemoFiles: FileList;

    uiDemoUseDialog: string;

    uiDemoUseTabular: boolean;
    uiDemoUseTitle: boolean;
    uiDemoUseClose: boolean;
    uiDemoUseFooter: boolean;
    uiDemoUseFrame: boolean;

    uiDemoUseWidth: number;
    uiDemoUseHeight: number;

}

const StoreKeys = [
    'uiDemoString',
    'uiDemoNumber',
    'uiDemoColor',
    'uiDemoCountry',
    'uiDemoDate',
    'uiDemoAllDisabled',
    'uiDemoBool',
    'uiDemoFiles',

    'uiDemoUseDialog',
    'uiDemoUseTabular',
    'uiDemoUseTitle',
    'uiDemoUseClose',
    'uiDemoUseFooter',
    'uiDemoUseFrame',

    'uiDemoUseWidth',
    'uiDemoUseHeight',

];

const COUNTRIES = [
    {name: 'Afghanistan', code: 'AF'},
    {name: 'Åland Islands', code: 'AX'},
    {name: 'Albania', code: 'AL'},
    {name: 'Algeria', code: 'DZ'},
    {name: 'American Samoa', code: 'AS'},
    {name: 'Andorra', code: 'AD'},
    {name: 'Angola', code: 'AO'},
    {name: 'Anguilla', code: 'AI'},
    {name: 'Antarctica', code: 'AQ'},
    {name: 'Antigua and Barbuda', code: 'AG'},
    {name: 'Argentina', code: 'AR'},
    {name: 'Armenia', code: 'AM'},
    {name: 'Aruba', code: 'AW'},
    {name: 'Australia', code: 'AU'},
    {name: 'Austria', code: 'AT'},
    {name: 'Azerbaijan', code: 'AZ'},
    {name: 'Bahamas', code: 'BS'},
    {name: 'Bahrain', code: 'BH'},
    {name: 'Bangladesh', code: 'BD'},
    {name: 'Barbados', code: 'BB'},
    {name: 'Belarus', code: 'BY'},
    {name: 'Belgium', code: 'BE'},
    {name: 'Belize', code: 'BZ'},
    {name: 'Benin', code: 'BJ'},
    {name: 'Bermuda', code: 'BM'},
    {name: 'Bhutan', code: 'BT'},
    {name: 'Bolivia', code: 'BO'},
    {name: 'Bosnia and Herzegovina', code: 'BA'},
    {name: 'Botswana', code: 'BW'},
    {name: 'Bouvet Island', code: 'BV'},
    {name: 'Brazil', code: 'BR'},
    {name: 'British Indian Ocean Territory', code: 'IO'},
    {name: 'Brunei Darussalam', code: 'BN'},
    {name: 'Bulgaria', code: 'BG'},
    {name: 'Burkina Faso', code: 'BF'},
    {name: 'Burundi', code: 'BI'},
    {name: 'Cambodia', code: 'KH'},
    {name: 'Cameroon', code: 'CM'},
    {name: 'Canada', code: 'CA'},
    {name: 'Cape Verde', code: 'CV'},
    {name: 'Cayman Islands', code: 'KY'},
    {name: 'Central African Republic', code: 'CF'},
    {name: 'Chad', code: 'TD'},
    {name: 'Chile', code: 'CL'},
    {name: 'China', code: 'CN'},
    {name: 'Christmas Island', code: 'CX'},
    {name: 'Cocos (Keeling) Islands', code: 'CC'},
    {name: 'Colombia', code: 'CO'},
    {name: 'Comoros', code: 'KM'},
    {name: 'Congo', code: 'CG'},
    {name: 'Congo, The Democratic Republic of the', code: 'CD'},
    {name: 'Cook Islands', code: 'CK'},
    {name: 'Costa Rica', code: 'CR'},
    {name: 'Cote D\'Ivoire', code: 'CI'},
    {name: 'Croatia', code: 'HR'},
    {name: 'Cuba', code: 'CU'},
    {name: 'Cyprus', code: 'CY'},
    {name: 'Czech Republic', code: 'CZ'},
    {name: 'Denmark', code: 'DK'},
    {name: 'Djibouti', code: 'DJ'},
    {name: 'Dominica', code: 'DM'},
    {name: 'Dominican Republic', code: 'DO'},
    {name: 'Ecuador', code: 'EC'},
    {name: 'Egypt', code: 'EG'},
    {name: 'El Salvador', code: 'SV'},
    {name: 'Equatorial Guinea', code: 'GQ'},
    {name: 'Eritrea', code: 'ER'},
    {name: 'Estonia', code: 'EE'},
    {name: 'Ethiopia', code: 'ET'},
    {name: 'Falkland Islands (Malvinas)', code: 'FK'},
    {name: 'Faroe Islands', code: 'FO'},
    {name: 'Fiji', code: 'FJ'},
    {name: 'Finland', code: 'FI'},
    {name: 'France', code: 'FR'},
    {name: 'French Guiana', code: 'GF'},
    {name: 'French Polynesia', code: 'PF'},
    {name: 'French Southern Territories', code: 'TF'},
    {name: 'Gabon', code: 'GA'},
    {name: 'Gambia', code: 'GM'},
    {name: 'Georgia', code: 'GE'},
    {name: 'Germany', code: 'DE'},
    {name: 'Ghana', code: 'GH'},
    {name: 'Gibraltar', code: 'GI'},
    {name: 'Greece', code: 'GR'},
    {name: 'Greenland', code: 'GL'},
    {name: 'Grenada', code: 'GD'},
    {name: 'Guadeloupe', code: 'GP'},
    {name: 'Guam', code: 'GU'},
    {name: 'Guatemala', code: 'GT'},
    {name: 'Guernsey', code: 'GG'},
    {name: 'Guinea', code: 'GN'},
    {name: 'Guinea-Bissau', code: 'GW'},
    {name: 'Guyana', code: 'GY'},
    {name: 'Haiti', code: 'HT'},
    {name: 'Heard Island and Mcdonald Islands', code: 'HM'},
    {name: 'Holy See (Vatican City State)', code: 'VA'},
    {name: 'Honduras', code: 'HN'},
    {name: 'Hong Kong', code: 'HK'},
    {name: 'Hungary', code: 'HU'},
    {name: 'Iceland', code: 'IS'},
    {name: 'India', code: 'IN'},
    {name: 'Indonesia', code: 'ID'},
    {name: 'Iran, Islamic Republic Of', code: 'IR'},
    {name: 'Iraq', code: 'IQ'},
    {name: 'Ireland', code: 'IE'},
    {name: 'Isle of Man', code: 'IM'},
    {name: 'Israel', code: 'IL'},
    {name: 'Italy', code: 'IT'},
    {name: 'Jamaica', code: 'JM'},
    {name: 'Japan', code: 'JP'},
    {name: 'Jersey', code: 'JE'},
    {name: 'Jordan', code: 'JO'},
    {name: 'Kazakhstan', code: 'KZ'},
    {name: 'Kenya', code: 'KE'},
    {name: 'Kiribati', code: 'KI'},
    {name: 'Korea, Democratic People\'S Republic of', code: 'KP'},
    {name: 'Korea, Republic of', code: 'KR'},
    {name: 'Kuwait', code: 'KW'},
    {name: 'Kyrgyzstan', code: 'KG'},
    {name: 'Lao People\'S Democratic Republic', code: 'LA'},
    {name: 'Latvia', code: 'LV'},
    {name: 'Lebanon', code: 'LB'},
    {name: 'Lesotho', code: 'LS'},
    {name: 'Liberia', code: 'LR'},
    {name: 'Libyan Arab Jamahiriya', code: 'LY'},
    {name: 'Liechtenstein', code: 'LI'},
    {name: 'Lithuania', code: 'LT'},
    {name: 'Luxembourg', code: 'LU'},
    {name: 'Macao', code: 'MO'},
    {name: 'Macedonia, The Former Yugoslav Republic of', code: 'MK'},
    {name: 'Madagascar', code: 'MG'},
    {name: 'Malawi', code: 'MW'},
    {name: 'Malaysia', code: 'MY'},
    {name: 'Maldives', code: 'MV'},
    {name: 'Mali', code: 'ML'},
    {name: 'Malta', code: 'MT'},
    {name: 'Marshall Islands', code: 'MH'},
    {name: 'Martinique', code: 'MQ'},
    {name: 'Mauritania', code: 'MR'},
    {name: 'Mauritius', code: 'MU'},
    {name: 'Mayotte', code: 'YT'},
    {name: 'Mexico', code: 'MX'},
    {name: 'Micronesia, Federated States of', code: 'FM'},
    {name: 'Moldova, Republic of', code: 'MD'},
    {name: 'Monaco', code: 'MC'},
    {name: 'Mongolia', code: 'MN'},
    {name: 'Montserrat', code: 'MS'},
    {name: 'Morocco', code: 'MA'},
    {name: 'Mozambique', code: 'MZ'},
    {name: 'Myanmar', code: 'MM'},
    {name: 'Namibia', code: 'NA'},
    {name: 'Nauru', code: 'NR'},
    {name: 'Nepal', code: 'NP'},
    {name: 'Netherlands', code: 'NL'},
    {name: 'Netherlands Antilles', code: 'AN'},
    {name: 'New Caledonia', code: 'NC'},
    {name: 'New Zealand', code: 'NZ'},
    {name: 'Nicaragua', code: 'NI'},
    {name: 'Niger', code: 'NE'},
    {name: 'Nigeria', code: 'NG'},
    {name: 'Niue', code: 'NU'},
    {name: 'Norfolk Island', code: 'NF'},
    {name: 'Northern Mariana Islands', code: 'MP'},
    {name: 'Norway', code: 'NO'},
    {name: 'Oman', code: 'OM'},
    {name: 'Pakistan', code: 'PK'},
    {name: 'Palau', code: 'PW'},
    {name: 'Palestinian Territory, Occupied', code: 'PS'},
    {name: 'Panama', code: 'PA'},
    {name: 'Papua New Guinea', code: 'PG'},
    {name: 'Paraguay', code: 'PY'},
    {name: 'Peru', code: 'PE'},
    {name: 'Philippines', code: 'PH'},
    {name: 'Pitcairn', code: 'PN'},
    {name: 'Poland', code: 'PL'},
    {name: 'Portugal', code: 'PT'},
    {name: 'Puerto Rico', code: 'PR'},
    {name: 'Qatar', code: 'QA'},
    {name: 'Reunion', code: 'RE'},
    {name: 'Romania', code: 'RO'},
    {name: 'Russian Federation', code: 'RU'},
    {name: 'RWANDA', code: 'RW'},
    {name: 'Saint Helena', code: 'SH'},
    {name: 'Saint Kitts and Nevis', code: 'KN'},
    {name: 'Saint Lucia', code: 'LC'},
    {name: 'Saint Pierre and Miquelon', code: 'PM'},
    {name: 'Saint Vincent and the Grenadines', code: 'VC'},
    {name: 'Samoa', code: 'WS'},
    {name: 'San Marino', code: 'SM'},
    {name: 'Sao Tome and Principe', code: 'ST'},
    {name: 'Saudi Arabia', code: 'SA'},
    {name: 'Senegal', code: 'SN'},
    {name: 'Serbia and Montenegro', code: 'CS'},
    {name: 'Seychelles', code: 'SC'},
    {name: 'Sierra Leone', code: 'SL'},
    {name: 'Singapore', code: 'SG'},
    {name: 'Slovakia', code: 'SK'},
    {name: 'Slovenia', code: 'SI'},
    {name: 'Solomon Islands', code: 'SB'},
    {name: 'Somalia', code: 'SO'},
    {name: 'South Africa', code: 'ZA'},
    {name: 'South Georgia and the South Sandwich Islands', code: 'GS'},
    {name: 'Spain', code: 'ES'},
    {name: 'Sri Lanka', code: 'LK'},
    {name: 'Sudan', code: 'SD'},
    {name: 'Suriname', code: 'SR'},
    {name: 'Svalbard and Jan Mayen', code: 'SJ'},
    {name: 'Swaziland', code: 'SZ'},
    {name: 'Sweden', code: 'SE'},
    {name: 'Switzerland', code: 'CH'},
    {name: 'Syrian Arab Republic', code: 'SY'},
    {name: 'Taiwan, Province of China', code: 'TW'},
    {name: 'Tajikistan', code: 'TJ'},
    {name: 'Tanzania, United Republic of', code: 'TZ'},
    {name: 'Thailand', code: 'TH'},
    {name: 'Timor-Leste', code: 'TL'},
    {name: 'Togo', code: 'TG'},
    {name: 'Tokelau', code: 'TK'},
    {name: 'Tonga', code: 'TO'},
    {name: 'Trinidad and Tobago', code: 'TT'},
    {name: 'Tunisia', code: 'TN'},
    {name: 'Turkey', code: 'TR'},
    {name: 'Turkmenistan', code: 'TM'},
    {name: 'Turks and Caicos Islands', code: 'TC'},
    {name: 'Tuvalu', code: 'TV'},
    {name: 'Uganda', code: 'UG'},
    {name: 'Ukraine', code: 'UA'},
    {name: 'United Arab Emirates', code: 'AE'},
    {name: 'United Kingdom', code: 'GB'},
    {name: 'United States', code: 'US'},
    {name: 'United States Minor Outlying Islands', code: 'UM'},
    {name: 'Uruguay', code: 'UY'},
    {name: 'Uzbekistan', code: 'UZ'},
    {name: 'Vanuatu', code: 'VU'},
    {name: 'Venezuela', code: 'VE'},
    {name: 'Viet Nam', code: 'VN'},
    {name: 'Virgin Islands, British', code: 'VG'},
    {name: 'Virgin Islands, U.S.', code: 'VI'},
    {name: 'Wallis and Futuna', code: 'WF'},
    {name: 'Western Sahara', code: 'EH'},
    {name: 'Yemen', code: 'YE'},
    {name: 'Zambia', code: 'ZM'},
    {name: 'Zimbabwe', code: 'ZW'}
].map(({name, code}) => ({text: name, value: code}));

class SmallForm extends gws.View<ViewProps> {
    render() {
        let bind = name => value => this.props.controller.update({[name]: value})

        return <Form tabular>
            <gws.ui.Select
                value={this.props.uiDemoCountry}
                label="select"
                items={COUNTRIES.slice(0, 5)}
                whenChanged={bind('uiDemoCountry')}
            />
            <gws.ui.TextInput
                value={this.props.uiDemoString}
                label="text input"
                whenChanged={bind('uiDemoString')}
            />
            <gws.ui.Group label="options">
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
                <gws.ui.Toggle
                    type="radio"
                    label="radio"
                    value={this.props.uiDemoBool}
                    whenChanged={bind('uiDemoBool')}
                />
            </gws.ui.Group>
            <gws.ui.ColorPicker
                value={this.props.uiDemoColor}
                label="Hintergrundfarbe"
                whenChanged={bind('uiDemoColor')}
            />
            <gws.ui.Group label="options" vertical>
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
            </gws.ui.Group>
        </Form>
    }
}


class FormDemo extends gws.View<ViewProps> {
    render() {
        let bind = name => value => this.props.controller.update({[name]: value})

        return <Form>
            <Row>
                <Cell flex>
                    <gws.ui.Select
                        value={this.props.uiDemoCountry}
                        label="select"
                        items={COUNTRIES.slice(0, 5)}
                        whenChanged={bind('uiDemoCountry')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.Select
                        value={this.props.uiDemoCountry}
                        label="search"
                        items={COUNTRIES}
                        withSearch
                        whenChanged={bind('uiDemoCountry')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.Select
                        value={this.props.uiDemoCountry}
                        label="combo"
                        withCombo
                        items={COUNTRIES}
                        whenChanged={bind('uiDemoCountry')}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.Select
                        value={this.props.uiDemoCountry}
                        label="disabled"
                        disabled
                        withCombo
                        items={COUNTRIES}
                        whenChanged={bind('uiDemoCountry')}
                    />
                </Cell>
            </Row>

            <Row>
                <Cell>
                    <gws.ui.DateInput
                        value={this.props.uiDemoDate}
                        label="date"
                        format={{
                            date: this.props.controller.app.localeData.dateFormatShort
                        }}
                        whenChanged={bind('uiDemoDate')}
                    />
                </Cell>
                <Cell>
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
                    <gws.ui.PasswordInput
                        value={this.props.uiDemoString}
                        label="password"
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
                        format={{
                            decimal: this.props.controller.app.localeData.numberDecimal,
                            group: this.props.controller.app.localeData.numberGroup
                        }}
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
                    <gws.ui.Group label="options">
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
                        <gws.ui.Toggle
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
                    </gws.ui.Group>
                </Cell>
            </Row>

        </Form>
    }
}

class SidebarBody extends gws.View<ViewProps> {

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
                            <gws.ui.Button
                                whenTouched={set('uiDemoUseDialog', 'form')}
                                label="form"/>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoUseDialog', 'alertError')}
                                label="error"/>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoUseDialog', 'alertInfo')}
                                label="info"/>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoUseDialog', 'popup')}
                                label="popup"/>
                        </Cell>
                    </Row>

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
                            <gws.ui.Toggle
                                type="checkbox"
                                label="title"
                                value={this.props.uiDemoUseTitle}
                                whenChanged={bind('uiDemoUseTitle')}
                            />
                            <gws.ui.Toggle
                                type="checkbox"
                                label="close"
                                value={this.props.uiDemoUseClose}
                                whenChanged={bind('uiDemoUseClose')}
                            />
                            <gws.ui.Toggle
                                type="checkbox"
                                label="footer"
                                value={this.props.uiDemoUseFooter}
                                whenChanged={bind('uiDemoUseFooter')}
                            />
                            <gws.ui.Toggle
                                type="checkbox"
                                label="frame"
                                value={this.props.uiDemoUseFrame}
                                whenChanged={bind('uiDemoUseFrame')}
                            />
                        </Cell>
                    </Row>
                    <Row>
                        <Cell>
                            <gws.ui.Button
                                whenTouched={set('uiDemoUseDialog', 'dialog')}
                                label="dialog"/>
                        </Cell>
                    </Row>

                </Form>

                <pre style={{padding: 10}}>
                    uiDemoString: {this.props.uiDemoString}<br/>
                    uiDemoNumber: {this.props.uiDemoNumber}<br/>
                    uiDemoColor: {this.props.uiDemoColor}<br/>
                    uiDemoCountry: {this.props.uiDemoCountry}<br/>
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
        let dm = this.props.uiDemoUseDialog;

        if (!dm)
            return null;

        let close = () => this.props.controller.update({
            uiDemoUseDialog: ''
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

        if (dm === 'form') {
            return <gws.ui.Dialog
                title='Form Controls'
                whenClosed={close}
                buttons={buttons}
            >
                <FormDemo {...this.props}/>
            </gws.ui.Dialog>
        }

        if (dm === 'dialog') {
            return <gws.ui.Dialog
                title={this.props.uiDemoUseTitle ? "Dialog Title" : null}
                whenClosed={this.props.uiDemoUseClose ? close : null}
                buttons={this.props.uiDemoUseFooter ? buttons : null}
                style={CENTER_BOX(this.props.uiDemoUseWidth, this.props.uiDemoUseHeight)}
                frame={this.props.uiDemoUseFrame ? '/chess.png' : null}
            >
                <SmallForm {...this.props}/>
            </gws.ui.Dialog>
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
            uiDemoCountry: 'DE',
            uiDemoDate: '2018-11-22',

            uiDemoUseDialog: 'form',

            uiDemoUseTabular: true,
            uiDemoUseTitle: true,
            uiDemoUseClose: true,
            uiDemoUseFooter: true,
            uiDemoUseFrame: false,

            uiDemoUseWidth: 580,
            uiDemoUseHeight: 580,
        })
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

