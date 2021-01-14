import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;


const MASTER = 'Shared.Tabedit';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as TabeditController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as TabeditController;
}

interface TabeditViewProps extends gws.types.ViewProps {
    controller: TabeditController;
    tabeditTables: Array<gws.api.TabeditTableProps>;
    tabeditData: gws.api.TabeditLoadDataResponse;
    tabeditTableUid: string;
    tabeditDate: string;
    tabeditDialogMode: string;
    tabeditPageSize: number;
    tabeditPage: number;
    tabeditDirtyFields: object;
    tabeditSelectRecord: number;
    tabeditError: boolean;
}

const TabeditStoreKeys = [
    'tabeditTables',
    'tabeditFeatures',
    'tabeditData',
    'tabeditDialogMode',
    'tabeditPageSize',
    'tabeditPage',
    'tabeditDirtyFields',
    'tabeditSelectRecord',
    'tabeditError',
];


class TabeditDialog extends gws.View<TabeditViewProps> {
    focusRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.focusRef = React.createRef();
    }

    close() {
        this.props.controller.update({tabeditDialogMode: ''});
    }

    message(mode) {
        switch (mode) {
            case 'success':
                return <div className="modTabeditFormPadding">
                    <p>{this.__('modTabeditDialogSuccess')}</p>
                </div>;

            case 'error':
                return <div className="modTabeditFormPadding">
                    <gws.ui.Error text={this.__('modTabeditDialogError')}/>
                </div>;
        }

    }

    componentDidMount() {
        console.log(this.focusRef.current)
        if (this.focusRef.current)
            gws.tools.nextTick(() => this.focusRef.current.focus());
    }

    componentDidUpdate(prevProps) {
        if (this.focusRef.current)
            gws.tools.nextTick(() => this.focusRef.current.focus());
    }


    tableDialog() {

        let data = this.props.tabeditData,
            len = data.records.length,
            pageSize = this.props.tabeditPageSize || 100,
            page = this.props.tabeditPage || 0,
            lastPage = Math.ceil(len / pageSize) - 1,
            dirtyFields = this.props.tabeditDirtyFields || {};

        if (page < 0)
            page = 0;
        if (page > lastPage)
            page = lastPage;

        let goTo = p => this.props.controller.update({
            tabeditPage: p,
            tabeditSelectRecord: -1,
        });

        let pager = (lastPage > 0) && <React.Fragment>
            <Cell>
                <gws.ui.Button
                    className="uiPagerFirst"
                    whenTouched={() => goTo(0)}
                    disabled={page == 0}
                />
            </Cell>
            <Cell>
                <gws.ui.Button
                    className="uiPagerPrev"
                    whenTouched={() => goTo(page - 1)}
                    disabled={page == 0}
                />
            </Cell>
            <Cell>
                <gws.ui.Button
                    className="uiPagerNext"
                    whenTouched={() => goTo(page + 1)}
                    disabled={page >= lastPage}
                />
            </Cell>
            <Cell>
                <gws.ui.Button
                    className="uiPagerLast"
                    whenTouched={() => goTo(lastPage)}
                    disabled={page >= lastPage}
                />
            </Cell>
        </React.Fragment>;

        let footer = <Row>
            {pager}
            <Cell>
                {this.props.tabeditError && <gws.ui.Error text={this.__('modTabeditDialogError')}/>}
            </Cell>
            <Cell flex/>
            <Cell>
                <gws.ui.Button
                    className="modTabeditButtonAdd"
                    whenTouched={() => this.props.controller.addRecord()}
                />
            </Cell>
            <Cell>
                <gws.ui.Button
                    className="modTabeditButtonSave"
                    disabled={gws.tools.empty(dirtyFields)}
                    whenTouched={() => this.props.controller.saveData()}
                />
            </Cell>
            <Cell>
                <gws.ui.Button
                    className="cmpButtonFormCancel"
                    whenTouched={() => this.close()}
                />
            </Cell>
        </Row>;

        let update = (nrec, ncol, value) => {
            let d = {...dirtyFields},
                key = nrec + '.' + ncol;

            if (value === data.records[nrec][ncol])
                delete d[key];
            else
                d[key] = value;

            this.props.controller.update({
                tabeditSelectRecord: -1,
                tabeditDirtyFields: d
            });
        };

        let getRow = nrow => {
            let nrec = page * pageSize + nrow,
                record = data.records[nrec];

            if (!record)
                return;

            return data.attributes.map((a, ncol) => {
                let key = nrec + '.' + ncol,
                    isDirty = key in dirtyFields,
                    val = isDirty ? dirtyFields[key] : record[ncol];

                if (!a.editable)
                    return String(val);

                if (a.type === gws.api.AttributeType.int)
                    return <gws.ui.NumberInput
                        value={val}
                        className={isDirty ? 'isDirty' : ''}
                        whenChanged={v => update(nrec, ncol, v)}
                    />;

                return <gws.ui.TextInput
                    value={val}
                    className={isDirty ? 'isDirty' : ''}
                    whenChanged={v => update(nrec, ncol, v)}
                />;
            });
        }

        let sel = this.props.tabeditSelectRecord;

        if (sel >= 0)
            sel -= page * pageSize;

        let table = <gws.ui.Table
            numRows={pageSize}
            getRow={getRow}
            fixedColumns={0}
            selectedRow={sel}
            headers={data.attributes.map(a => a.title)}
        />;

        return <gws.ui.Dialog
            className="modTabeditDialog"
            title={this.props.controller.tableTitle()}
            whenClosed={() => this.close()}
            footer={footer}
        >{table}</gws.ui.Dialog>
    }

    render() {
        let mode = this.props.tabeditDialogMode;
        if (!mode)
            return null;

        if (mode === 'open')
            return this.tableDialog()


        if (mode === 'wait') {
            return <gws.ui.Dialog
                className="modTabeditDialog"
                title={this.props.controller.tableTitle()}
            >
                <gws.ui.Text
                    className="modTabeditDialogLoading"
                    content={this.__('modTabeditDialogLoading')}
                />
                <gws.ui.Loader/>
            </gws.ui.Dialog>
        }

        return <gws.ui.Dialog
            className='modTabeditSmallDialog'
            whenClosed={() => this.close()}
        >{this.message(mode)}</gws.ui.Dialog>

    }
}

class TabeditSidebarView extends gws.View<TabeditViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modTabeditSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {
                    this.props.tabeditTables.map(t => <Row key={t.uid}>
                        <Cell>
                            <gws.ui.Button className='modTabeditListButton'/>
                        </Cell>
                        <Cell>
                            <gws.ui.Touchable
                                className="modTabeditListTitle"
                                whenTouched={() => cc.startEdit(t.uid)}
                            >{t.title}</gws.ui.Touchable>
                        </Cell>
                    </Row>)
                }

            </sidebar.TabBody>

            <sidebar.TabFooter>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class TabeditSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modTabeditSidebarIcon';

    get tooltip() {
        return this.__('modTabeditSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(TabeditSidebarView, TabeditStoreKeys));
    }

}

const DEFAULT_PAGE_SIZE = 100;

class TabeditController extends gws.Controller {
    uid = MASTER;

    async init() {
        let res = await this.app.server.tabeditGetTables({});
        this.update({
            tabeditTables: res.tables,
            tabeditPageSize: DEFAULT_PAGE_SIZE,
            tabeditPage: 0,
        });
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(TabeditDialog, TabeditStoreKeys));
    }

    tableTitle() {
        let uid = this.getValue('tabeditTableUid');
        for (let tbl of this.getValue('tabeditTables')) {
            if (tbl.uid === uid)
                return tbl.title;
        }
        return '';
    }

    async startEdit(tableUid) {
        this.update({
            tabeditDialogMode: 'wait',
        })

        console.time('TABEDIT: load');
        let res = await this.app.server.tabeditLoadData({tableUid});
        console.timeEnd('TABEDIT: load');

        this.update({
            tabeditTableUid: tableUid,
            tabeditData: res,
            tabeditDialogMode: 'open',
            tabeditPage: 0,
            tabeditSelectRecord: -1,
            tabeditError: false,
        })
    }

    addRecord() {
        let data = this.getValue('tabeditData'),
            rec = [];

        for (let a of data.attributes) {
            rec.push('');
        }

        this.update({
            tabeditData: {
                ...data,
                records: data.records.concat([rec]),
            },
            tabeditPage: 1e10,
            tabeditSelectRecord: data.records.length,
        })
    }

    async saveData() {
        let data = this.getValue('tabeditData'),
            dirtyFields = this.getValue('tabeditDirtyFields') || {},
            recMap = {};

        for (let [key, val] of gws.tools.entries(dirtyFields)) {
            let [nrec, ncol] = key.split('.');
            if (!recMap[nrec])
                recMap[nrec] = data.records[nrec];
            recMap[nrec][ncol] = val;
        }

        let params: gws.api.TabeditSaveDataParams = {
            tableUid: this.getValue('tabeditTableUid'),
            attributes: data.attributes,
            records: Object.values(recMap),
        };

        let res = await this.app.server.tabeditSaveData(params);

        if (res.error) {
            this.update({
                tabeditError: true,
            });
        } else {
            this.update({
                tabeditError: false,
                tabeditDirtyFields: {},
            });
        }
    }

}

export const tags = {
    [MASTER]: TabeditController,
    'Sidebar.Tabedit': TabeditSidebar,
};
