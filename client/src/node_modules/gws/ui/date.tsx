import * as React from 'react';
import * as moment from 'moment';

import * as base from './base';
import * as util from './util';

import {Touchable} from './button';
import {NumberInputBox} from './number';
import {Row, Cell} from './layout';


// like en_US but with Monday=0

const DEFAULT_DATE_FORMAT = {
    dateFormatLong: 'yyyy-MM-dd',
    dateFormatMedium: 'yyyy-MM-dd',
    dateFormatShort: 'yyyy-MM-dd',
    dayNamesLong: 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split(' '),
    dayNamesShort: 'Mo Tu We Th Fr Sa Su'.split(' '),
    dayNamesNarrow: 'M T W T F S S'.split(' '),
    firstWeekDay: 0,
    monthNamesLong: 'January February March April May June July August September October November December'.split(' '),
    monthNamesShort: 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split(' '),
    monthNamesNarrow: 'J F M A M J J A S O N D'.split(' '),
};

interface DateInputProps extends base.InputProps<string> {
    minValue?: string;
    maxValue?: string;
    withClear?: boolean;
    whenEntered?: (value: string) => void;
    locale?: util.Locale;
}

export class DateInput extends base.Control<DateInputProps> {
    render() {
        let dmy = util.iso2dmy(this.props.value),
            lo = this.props.locale || (DEFAULT_DATE_FORMAT as util.Locale);

        let inputProps = {
            className: 'uiRawInput',
            disabled: this.props.disabled,
            onClick: e => this.setOpen(true),
            placeholder: this.props.placeholder || '',
            ref: this.focusRef,
            tabIndex: 0,
            title: this.props.tooltip || '',
            type: 'text',
            value: dmy ? util.formatDate(dmy, lo.dateFormatShort, lo) : '',
            readOnly: true,
        };


        return <base.Content of={this} withClass="uiDateInput">
            <base.Box>
                <input {...inputProps}/>
                <Cell flex/>
                {this.props.withClear && <Cell><Touchable
                    className={'uiClearButton' + (util.empty(this.props.value) ? ' isHidden' : '')}
                    whenTouched={() => this.clear()}/></Cell>
                }
                <Touchable
                    className="uiDateInputCalendarButton"
                    whenTouched={() => this.toggle()}/>
            </base.Box>
            <base.DropDown>
                <DateDropDown
                    value={this.props.value}
                    locale={lo}
                    focusRef={this.focusRef}
                    minValue={this.props.minValue || '0000-00-00'}
                    maxValue={this.props.maxValue || '9999-99-99'}
                    whenChanged={v => this.whenDropDownChanged(v)}
                />
            </base.DropDown>
        </base.Content>
    }

    protected whenDropDownChanged(v: string) {
        if (this.props.whenChanged)
            this.props.whenChanged(v);
    }

    protected onChange(e: React.SyntheticEvent<HTMLInputElement>) {
        if (this.props.whenChanged)
            this.props.whenChanged(e.currentTarget.value);
    }


    protected clear() {
        this.grabFocus();
        if (this.props.whenChanged)
            this.props.whenChanged('');
    }

    protected toggle() {
        this.grabFocus();
        this.toggleOpen();
    }
}


interface DateDropDownProps extends base.ControlProps {
    value: string;
    locale: util.Locale;
    minValue: string;
    maxValue: string;
    whenChanged: (value: string) => void;
}

interface DateDropDownState extends base.ControlState {
    dmy: util.DMY;
}


class DateDropDown extends base.Control<DateDropDownProps, DateDropDownState> {
    constructor(props) {
        super(props);
        this.state.dmy = util.iso2dmy(this.props.value) || util.date2dmy(new Date());
    }

    render() {
        return <div className="uiDropDownContent">
            <Row className="uiDateInputCalendarHead">
                <Cell>
                    <Touchable className="uiLeftButton" whenTouched={() => this.update('m-1')}/>
                </Cell>
                <Cell>
                    {this.props.locale.monthNamesShort[this.state.dmy.m - 1]}
                </Cell>
                <Cell>
                    <Touchable className="uiRightButton" whenTouched={() => this.update('m+1')}/>
                </Cell>
                <Cell flex/>
                <Cell>
                    <Touchable className="uiLeftButton" whenTouched={() => this.update('y-1')}/>
                </Cell>
                <Cell>
                    {this.state.dmy.y}
                </Cell>
                <Cell>
                    <Touchable className="uiRightButton" whenTouched={() => this.update('y+1')}/>
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <table className="uiDateInputCalendarTable">
                        <tbody>{this.table()}</tbody>
                    </table>
                </Cell>
            </Row>
        </div>
    }

    protected table() {
        let dmy = this.state.dmy,
            today = util.date2iso(),
            mat = calendarMatrix(dmy, this.props.locale.firstWeekDay),
            table = [];

        for (let r = 0; r < mat.length; r++) {
            let row = [];
            for (let c = 0; c < 7; c++) {
                row.push(this.tableCell(c, mat[r][c], today))
            }
            table.push(<tr key={r}>{row}</tr>)
        }
        return table;
    }

    protected tableCell(c, v, today) {
        if (!v) {
            return <td key={c}/>;
        }

        let [day, iso] = v;

        let ok = this.props.minValue <= iso && iso <= this.props.maxValue;

        let cls = util.className(
            'hasContent',
            !ok && 'isDisabled',
            (iso === today) && 'uiDateInputIsToday',
            (iso === this.props.value) && 'isSelected',
        );

        if (!ok) {
            return <td className={cls} key={c}>{day}</td>;
        }

        return <td className={cls} key={c}>
            <Touchable whenTouched={() => this.props.whenChanged(iso)}>
                {day}
            </Touchable>
        </td>
    }

    protected update(opt) {
        let dmy = {...this.state.dmy};

        this.grabFocus();

        switch (opt) {
            case 'm-1':
                if (dmy.m === 1) {
                    dmy.m = 12;
                    dmy.y--;
                } else {
                    dmy.m--;
                }
                break;
            case 'm+1':
                if (dmy.m === 12) {
                    dmy.m = 1;
                    dmy.y++;
                } else {
                    dmy.m++;
                }
                break;
            case 'y-1':
                dmy.y--;
                break;
            case 'y+1':
                dmy.y++;
                break;

        }

        this.setState({...this.state, dmy});
    }
}

function calendarMatrix(dmy, firstWeekDay) {
    let date = new Date(dmy.y, dmy.m - 1, 1, 12, 0);
    let mat = [];
    let r = -1;

    for (let d = 1; d <= 31; d++) {
        date.setDate(d);
        if (date.getMonth() !== dmy.m - 1)
            break;
        let c = date.getDay() - firstWeekDay - 1;
        if (c < 0)
            c += 7;
        if (r < 0 || c === 0) {
            mat.push([]);
            r++;
        }
        let iso = util.dmy2iso({d, m: dmy.m, y: dmy.y})
        mat[r][c] = [d, iso];
    }

    return mat;
}

