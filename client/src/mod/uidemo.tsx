import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from './sidebar';

interface P extends gws.types.ViewProps {
    controller: UIDemoController;
    uiDemoNumber: string;
    uiDemoCountry: string;
    uiDemoString: string;
    uiDemoString2: string;
    x1: boolean;
    x2: boolean;
    x3: boolean;
}

const countries = JSON.parse(
    '[{"value":"AF","text":"Afghanistan"},{"value":"AX","text":"Aland Islands"},{"value":"AL","text":"Albania"},{"value":"DZ","text":"Algeria"},{"value":"AS","text":"American Samoa"},{"value":"AD","text":"Andorra"},{"value":"AO","text":"Angola"},{"value":"AI","text":"Anguilla"},{"value":"AQ","text":"Antarctica"},{"value":"AG","text":"Antigua and Barbuda"},{"value":"AR","text":"Argentina"},{"value":"AM","text":"Armenia"},{"value":"AW","text":"Aruba"},{"value":"AU","text":"Australia"},{"value":"AT","text":"Austria"},{"value":"AZ","text":"Azerbaijan"},{"value":"BS","text":"Bahamas"},{"value":"BH","text":"Bahrain"},{"value":"BD","text":"Bangladesh"},{"value":"BB","text":"Barbados"},{"value":"BY","text":"Belarus"},{"value":"BE","text":"Belgium"},{"value":"BZ","text":"Belize"},{"value":"BJ","text":"Benin"},{"value":"BM","text":"Bermuda"},{"value":"BT","text":"Bhutan"},{"value":"BO","text":"Bolivia"},{"value":"BA","text":"Bosnia and Herzegovina"},{"value":"BW","text":"Botswana"},{"value":"BV","text":"Bouvet Island"},{"value":"BR","text":"Brazil"},{"value":"VG","text":"British Virgin Islands"},{"value":"IO","text":"British Indian Ocean Territory"},{"value":"BN","text":"Brunei Darussalam"},{"value":"BG","text":"Bulgaria"},{"value":"BF","text":"Burkina Faso"},{"value":"BI","text":"Burundi"},{"value":"KH","text":"Cambodia"},{"value":"CM","text":"Cameroon"},{"value":"CA","text":"Canada"},{"value":"CV","text":"Cape Verde"},{"value":"KY","text":"Cayman Islands"},{"value":"CF","text":"Central African Republic"},{"value":"TD","text":"Chad"},{"value":"CL","text":"Chile"},{"value":"CN","text":"China"},{"value":"HK","text":"Hong Kong, SAR China"},{"value":"MO","text":"Macao, SAR China"},{"value":"CX","text":"Christmas Island"},{"value":"CC","text":"Cocos (Keeling) Islands"},{"value":"CO","text":"Colombia"},{"value":"KM","text":"Comoros"},{"value":"CG","text":"Congo (Brazzaville)"},{"value":"CD","text":"Congo, (Kinshasa)"},{"value":"CK","text":"Cook Islands"},{"value":"CR","text":"Costa Rica"},{"value":"CI","text":"Côte d\'Ivoire"},{"value":"HR","text":"Croatia"},{"value":"CU","text":"Cuba"},{"value":"CY","text":"Cyprus"},{"value":"CZ","text":"Czech Republic"},{"value":"DK","text":"Denmark"},{"value":"DJ","text":"Djibouti"},{"value":"DM","text":"Dominica"},{"value":"DO","text":"Dominican Republic"},{"value":"EC","text":"Ecuador"},{"value":"EG","text":"Egypt"},{"value":"SV","text":"El Salvador"},{"value":"GQ","text":"Equatorial Guinea"},{"value":"ER","text":"Eritrea"},{"value":"EE","text":"Estonia"},{"value":"ET","text":"Ethiopia"},{"value":"FK","text":"Falkland Islands (Malvinas)"},{"value":"FO","text":"Faroe Islands"},{"value":"FJ","text":"Fiji"},{"value":"FI","text":"Finland"},{"value":"FR","text":"France"},{"value":"GF","text":"French Guiana"},{"value":"PF","text":"French Polynesia"},{"value":"TF","text":"French Southern Territories"},{"value":"GA","text":"Gabon"},{"value":"GM","text":"Gambia"},{"value":"GE","text":"Georgia"},{"value":"DE","text":"Germany"},{"value":"GH","text":"Ghana"},{"value":"GI","text":"Gibraltar"},{"value":"GR","text":"Greece"},{"value":"GL","text":"Greenland"},{"value":"GD","text":"Grenada"},{"value":"GP","text":"Guadeloupe"},{"value":"GU","text":"Guam"},{"value":"GT","text":"Guatemala"},{"value":"GG","text":"Guernsey"},{"value":"GN","text":"Guinea"},{"value":"GW","text":"Guinea-Bissau"},{"value":"GY","text":"Guyana"},{"value":"HT","text":"Haiti"},{"value":"HM","text":"Heard and Mcdonald Islands"},{"value":"VA","text":"Holy See (Vatican City State)"},{"value":"HN","text":"Honduras"},{"value":"HU","text":"Hungary"},{"value":"IS","text":"Iceland"},{"value":"IN","text":"India"},{"value":"ID","text":"Indonesia"},{"value":"IR","text":"Iran, Islamic Republic of"},{"value":"IQ","text":"Iraq"},{"value":"IE","text":"Ireland"},{"value":"IM","text":"Isle of Man"},{"value":"IL","text":"Israel"},{"value":"IT","text":"Italy"},{"value":"JM","text":"Jamaica"},{"value":"JP","text":"Japan"},{"value":"JE","text":"Jersey"},{"value":"JO","text":"Jordan"},{"value":"KZ","text":"Kazakhstan"},{"value":"KE","text":"Kenya"},{"value":"KI","text":"Kiribati"},{"value":"KP","text":"Korea (North)"},{"value":"KR","text":"Korea (South)"},{"value":"KW","text":"Kuwait"},{"value":"KG","text":"Kyrgyzstan"},{"value":"LA","text":"Lao PDR"},{"value":"LV","text":"Latvia"},{"value":"LB","text":"Lebanon"},{"value":"LS","text":"Lesotho"},{"value":"LR","text":"Liberia"},{"value":"LY","text":"Libya"},{"value":"LI","text":"Liechtenstein"},{"value":"LT","text":"Lithuania"},{"value":"LU","text":"Luxembourg"},{"value":"MK","text":"Macedonia, Republic of"},{"value":"MG","text":"Madagascar"},{"value":"MW","text":"Malawi"},{"value":"MY","text":"Malaysia"},{"value":"MV","text":"Maldives"},{"value":"ML","text":"Mali"},{"value":"MT","text":"Malta"},{"value":"MH","text":"Marshall Islands"},{"value":"MQ","text":"Martinique"},{"value":"MR","text":"Mauritania"},{"value":"MU","text":"Mauritius"},{"value":"YT","text":"Mayotte"},{"value":"MX","text":"Mexico"},{"value":"FM","text":"Micronesia, Federated States of"},{"value":"MD","text":"Moldova"},{"value":"MC","text":"Monaco"},{"value":"MN","text":"Mongolia"},{"value":"ME","text":"Montenegro"},{"value":"MS","text":"Montserrat"},{"value":"MA","text":"Morocco"},{"value":"MZ","text":"Mozambique"},{"value":"MM","text":"Myanmar"},{"value":"NA","text":"Namibia"},{"value":"NR","text":"Nauru"},{"value":"NP","text":"Nepal"},{"value":"NL","text":"Netherlands"},{"value":"AN","text":"Netherlands Antilles"},{"value":"NC","text":"New Caledonia"},{"value":"NZ","text":"New Zealand"},{"value":"NI","text":"Nicaragua"},{"value":"NE","text":"Niger"},{"value":"NG","text":"Nigeria"},{"value":"NU","text":"Niue"},{"value":"NF","text":"Norfolk Island"},{"value":"MP","text":"Northern Mariana Islands"},{"value":"NO","text":"Norway"},{"value":"OM","text":"Oman"},{"value":"PK","text":"Pakistan"},{"value":"PW","text":"Palau"},{"value":"PS","text":"Palestinian Territory"},{"value":"PA","text":"Panama"},{"value":"PG","text":"Papua New Guinea"},{"value":"PY","text":"Paraguay"},{"value":"PE","text":"Peru"},{"value":"PH","text":"Philippines"},{"value":"PN","text":"Pitcairn"},{"value":"PL","text":"Poland"},{"value":"PT","text":"Portugal"},{"value":"PR","text":"Puerto Rico"},{"value":"QA","text":"Qatar"},{"value":"RE","text":"Réunion"},{"value":"RO","text":"Romania"},{"value":"RU","text":"Russian Federation"},{"value":"RW","text":"Rwanda"},{"value":"BL","text":"Saint-Barthélemy"},{"value":"SH","text":"Saint Helena"},{"value":"KN","text":"Saint Kitts and Nevis"},{"value":"LC","text":"Saint Lucia"},{"value":"MF","text":"Saint-Martin (French part)"},{"value":"PM","text":"Saint Pierre and Miquelon"},{"value":"VC","text":"Saint Vincent and Grenadines"},{"value":"WS","text":"Samoa"},{"value":"SM","text":"San Marino"},{"value":"ST","text":"Sao Tome and Principe"},{"value":"SA","text":"Saudi Arabia"},{"value":"SN","text":"Senegal"},{"value":"RS","text":"Serbia"},{"value":"SC","text":"Seychelles"},{"value":"SL","text":"Sierra Leone"},{"value":"SG","text":"Singapore"},{"value":"SK","text":"Slovakia"},{"value":"SI","text":"Slovenia"},{"value":"SB","text":"Solomon Islands"},{"value":"SO","text":"Somalia"},{"value":"ZA","text":"South Africa"},{"value":"GS","text":"South Georgia and the South Sandwich Islands"},{"value":"SS","text":"South Sudan"},{"value":"ES","text":"Spain"},{"value":"LK","text":"Sri Lanka"},{"value":"SD","text":"Sudan"},{"value":"SR","text":"Suriname"},{"value":"SJ","text":"Svalbard and Jan Mayen Islands"},{"value":"SZ","text":"Swaziland"},{"value":"SE","text":"Sweden"},{"value":"CH","text":"Switzerland"},{"value":"SY","text":"Syrian Arab Republic (Syria)"},{"value":"TW","text":"Taiwan, Republic of China"},{"value":"TJ","text":"Tajikistan"},{"value":"TZ","text":"Tanzania, United Republic of"},{"value":"TH","text":"Thailand"},{"value":"TL","text":"Timor-Leste"},{"value":"TG","text":"Togo"},{"value":"TK","text":"Tokelau"},{"value":"TO","text":"Tonga"},{"value":"TT","text":"Trinidad and Tobago"},{"value":"TN","text":"Tunisia"},{"value":"TR","text":"Turkey"},{"value":"TM","text":"Turkmenistan"},{"value":"TC","text":"Turks and Caicos Islands"},{"value":"TV","text":"Tuvalu"},{"value":"UG","text":"Uganda"},{"value":"UA","text":"Ukraine"},{"value":"AE","text":"United Arab Emirates"},{"value":"GB","text":"United Kingdom"},{"value":"US","text":"United States of America"},{"value":"UM","text":"US Minor Outlying Islands"},{"value":"UY","text":"Uruguay"},{"value":"UZ","text":"Uzbekistan"},{"value":"VU","text":"Vanuatu"},{"value":"VE","text":"Venezuela (Bolivarian Republic)"},{"value":"VN","text":"Viet Nam"},{"value":"VI","text":"Virgin Islands, US"},{"value":"WF","text":"Wallis and Futuna Islands"},{"value":"EH","text":"Western Sahara"},{"value":"YE","text":"Yemen"},{"value":"ZM","text":"Zambia"},{"value":"ZW","text":"Zimbabwe"}]'
);

const colors = [
    '#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#ef5350', '#f44336', '#e53935', '#d32f2f', '#c62828', '#b71c1c', '#ff8a80', '#ff5252', '#ff1744', '#d50000', '#fce4ec', '#f8bbd0', '#f48fb1', '#f06292', '#ec407a', '#e91e63', '#d81b60', '#c2185b', '#ad1457', '#880e4f', '#ff80ab', '#ff4081', '#f50057', '#c51162', '#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0', '#8e24aa', '#7b1fa2', '#6a1b9a', '#4a148c', '#ea80fc', '#e040fb', '#d500f9', '#aa00ff', '#ede7f6', '#d1c4e9', '#b39ddb', '#9575cd', '#7e57c2', '#673ab7', '#5e35b1', '#512da8', '#4527a0', '#311b92', '#b388ff', '#7c4dff', '#651fff', '#6200ea', '#e8eaf6', '#c5cae9', '#9fa8da', '#7986cb', '#5c6bc0', '#3f51b5', '#3949ab', '#303f9f', '#283593', '#1a237e', '#8c9eff', '#536dfe', '#3d5afe', '#304ffe', '#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', '#2196f3', '#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#82b1ff', '#448aff', '#2979ff', '#2962ff', '#e1f5fe', '#b3e5fc', '#81d4fa', '#4fc3f7', '#29b6f6', '#03a9f4', '#039be5', '#0288d1', '#0277bd', '#01579b', '#80d8ff', '#40c4ff', '#00b0ff', '#0091ea', '#e0f7fa', '#b2ebf2', '#80deea', '#4dd0e1', '#26c6da', '#00bcd4', '#00acc1', '#0097a7', '#00838f', '#006064', '#84ffff', '#18ffff', '#00e5ff', '#00b8d4', '#e0f2f1', '#b2dfdb', '#80cbc4', '#4db6ac', '#26a69a', '#009688', '#00897b', '#00796b', '#00695c', '#004d40', '#a7ffeb', '#64ffda', '#1de9b6', '#00bfa5', '#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50', '#43a047', '#388e3c', '#2e7d32', '#1b5e20', '#b9f6ca', '#69f0ae', '#00e676', '#00c853', '#f1f8e9', '#dcedc8', '#c5e1a5', '#aed581', '#9ccc65', '#8bc34a', '#7cb342', '#689f38', '#558b2f', '#33691e', '#ccff90', '#b2ff59', '#76ff03', '#64dd17', '#f9fbe7', '#f0f4c3', '#e6ee9c', '#dce775', '#d4e157', '#cddc39', '#c0ca33', '#afb42b', '#9e9d24', '#827717', '#f4ff81', '#eeff41', '#c6ff00', '#aeea00', '#fffde7', '#fff9c4', '#fff59d', '#fff176', '#ffee58', '#ffeb3b', '#fdd835', '#fbc02d', '#f9a825', '#f57f17', '#ffff8d', '#ffff00', '#ffea00', '#ffd600', '#fff8e1', '#ffecb3', '#ffe082', '#ffd54f', '#ffca28', '#ffc107', '#ffb300', '#ffa000', '#ff8f00', '#ff6f00', '#ffe57f', '#ffd740', '#ffc400', '#ffab00', '#fff3e0', '#ffe0b2', '#ffcc80', '#ffb74d', '#ffa726', '#ff9800', '#fb8c00', '#f57c00', '#ef6c00', '#e65100', '#ffd180', '#ffab40', '#ff9100', '#ff6d00', '#fbe9e7', '#ffccbc', '#ffab91', '#ff8a65', '#ff7043', '#ff5722', '#f4511e', '#e64a19', '#d84315', '#bf360c', '#ff9e80', '#ff6e40', '#ff3d00', '#dd2c00', '#efebe9', '#d7ccc8', '#bcaaa4', '#a1887f', '#8d6e63', '#795548', '#6d4c41', '#5d4037', '#4e342e', '#3e2723', '#eceff1', '#cfd8dc', '#b0bec5', '#90a4ae', '#78909c', '#607d8b', '#546e7a', '#455a64', '#37474f', '#263238', '#fafafa', '#f5f5f5', '#eeeeee', '#e0e0e0', '#bdbdbd', '#9e9e9e', '#757575', '#616161', '#424242', '#212121', '#000000', '#ffffff',
];

const long = `
London. Michaelmas term lately over, and the Lord Chancellor sitting in Lincoln's Inn Hall. 
Implacable November weather. 
As much mud in the streets as if the waters had but newly retired from the face of the earth, 
and it would not be wonderful to meet a Megalosaurus, 
forty feet long or so, waddling like an elephantine lizard up Holborn Hill. 
`;

const countries2 = countries.slice(0, 5)

class DemoBody extends gws.View<P> {
    render() {
        let head = s => <div style={{fontWeight: 800}}>{s}</div>;
        let upd = args => this.props.controller.update(args);
        let {Form, Row, Cell, Divider} = gws.ui.Layout;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content="UIDemo"/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.ui.TextArea
                                label='Area'
                                height={80}
                                value={this.props.uiDemoString2}
                                whenChanged={value => upd({uiDemoString2: value})}
                            />
                        </Cell>
                    </Row>

                    <Row>
                        <Cell flex>
                            <gws.ui.TextInput
                                label='Input'
                                value={this.props.uiDemoString}
                                whenChanged={value => upd({uiDemoString: value})}
                            />
                        </Cell>
                    </Row>

                    <Row>
                        <Cell flex>
                            <gws.ui.Select
                                label="Normal select"
                                items={countries}
                                value={this.props.uiDemoCountry}
                                placeholder="Country"
                                whenChanged={value => upd({uiDemoCountry: value})}
                            />
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex>
                            <gws.ui.ColorSelect
                                label="Color select"
                                items={colors}
                                value={this.props.uiDemoString}
                                whenChanged={value => upd({uiDemoString: value})}
                            />
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex>
                            <gws.ui.Select
                                label="Select with search"
                                items={countries}
                                value={this.props.uiDemoCountry}
                                placeholder="Country"
                                whenChanged={value => upd({uiDemoCountry: value})}
                                withSearch
                            />
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex>
                            <gws.ui.Select
                                label="Select up"
                                up={true}
                                items={countries}
                                value={this.props.uiDemoCountry}
                                placeholder="Country"
                                whenChanged={value => upd({uiDemoCountry: value})}
                            />
                        </Cell>
                    </Row>
                </Form>
                <Divider/>
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.ui.Toggle label="Option 1" value={this.props.x1}
                                             whenChanged={value => upd({x1: value})}/>
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex>
                            <gws.ui.Toggle label="Option 2" value={this.props.x2}
                                             whenChanged={value => upd({x2: value})}/>
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex>
                            <gws.ui.Toggle label="Option 3" value={this.props.x3}
                                             whenChanged={value => upd({x3: value})}/>
                        </Cell>
                    </Row>
                    <Row last>
                        <Cell flex>
                            <gws.ui.Toggle
                                type="radio" right label="Option 1" value={this.props.x1}
                                whenChanged={value => upd({x1: value})}/>
                        </Cell>
                        <Cell flex>
                            <gws.ui.Toggle
                                type="radio" right label="Option 2" value={this.props.x2}
                                whenChanged={value => upd({x2: value})}/>
                        </Cell>
                        <Cell flex>
                            <gws.ui.Toggle
                                type="radio" right label="Option 3" value={this.props.x3}
                                whenChanged={value => upd({x3: value})}/>
                        </Cell>
                    </Row>

                    <Row>
                        <Cell width={100}>
                            <gws.ui.TextButton
                                tooltip='Primary button'
                                primary
                                whenTouched={() => upd({
                                    modalContent: <div>Hey modal dialog</div>
                                })}>Primary
                            </gws.ui.TextButton>
                        </Cell>

                        <Cell width={100}>
                            <gws.ui.TextButton
                                tooltip='This is a default button'
                                whenTouched={() => upd({uiDemoString: 'default touched!'})}>Default
                            </gws.ui.TextButton>
                        </Cell>
                    </Row>

                </Form>

                <Divider/>

                <Row>
                    <Cell flex>
                        <gws.ui.Slider
                            label="Slider"
                            minValue={0}
                            maxValue={100}
                            value={Number(this.props.uiDemoNumber) || 0}
                            whenChanged={value => upd({uiDemoNumber: value})}
                        />
                    </Cell>
                </Row>
                <Row last>
                    <Cell flex>
                        <gws.ui.Progress
                            label={(this.props.uiDemoNumber || '0') + '% done'}
                            value={Number(this.props.uiDemoNumber) || 0}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.Loader/>
                    </Cell>
                </Row>

                <Divider/>

                <Row>
                    <Cell flex>
                        <gws.ui.Error text="Error message"/>
                    </Cell>
                </Row>
                <Row last>
                    <Cell flex>
                        <gws.ui.Error
                            text="Error message"
                            longText="A software bug is an error, flaw, failure or fault in a computer program or system that causes it to produce an incorrect or unexpected result"
                        />
                    </Cell>
                </Row>

                <Divider/>

            </sidebar.TabBody>

            <sidebar.TabFooter>


                <p>number={this.props.uiDemoNumber}</p>
                <p>string={this.props.uiDemoString}</p>
                <p>country={this.props.uiDemoCountry}</p>


            </sidebar.TabFooter>

        </sidebar.Tab>
    }

}

class UIDemoController extends gws.Controller implements gws.types.ISidebarItem {

    features = [];

    async init() {
        this.update({
            uiDemoNumber: 25,
            uiDemoCountry: null,
            uiDemoString: 'Hello',
            uiDemoString2: long.trim(),
        })
    }

    get iconClass() {
        return 'modLayersSidebarIcon';
    }

    get tooltip() {
        return this.__('modLayersTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(DemoBody, [
                'uiDemoNumber', 'uiDemoCountry', 'uiDemoString', 'uiDemoString2',
                'x1', 'x2', 'x3'
            ]));
    }

}

export const tags = {
    'Sidebar.UIDemo': UIDemoController,
};

