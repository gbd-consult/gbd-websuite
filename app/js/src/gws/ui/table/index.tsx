import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

import {Text} from '../text';


interface TableProps {
    getRow: (r: number) => Array<React.ReactNode>;
    headers?: Array<Array<string | React.ReactNode>>;
    footers?: Array<Array<string | React.ReactNode>>;
    fixedLeftColumn?: boolean;
    fixedRightColumn?: boolean;
    numRows?: number;
    selectedRow?: number;
}


export class Table extends base.Pure<TableProps> {
    topRef: React.RefObject<any>;
    leftRef: React.RefObject<any>;
    bodyRef: React.RefObject<any>;
    focusRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.topRef = React.createRef();
        this.leftRef = React.createRef();
        this.bodyRef = React.createRef();
        this.focusRef = React.createRef();
    }

    componentDidMount() {
        this.setFocus();
    }

    componentDidUpdate(prevProps) {
        this.setFocus();
    }

    setFocus() {
        let row = this.focusRef.current;
        if (row) {
            let inp = row.querySelector('input');
            if (inp)
                util.nextTick(() => inp.focus());
        }
    }

    cell(value) {
        if (typeof value === "string") {
            return <Text content={value}/>
        }
        return value
    }

    renderRows(rows) {
        return rows.map((row, r) =>
            <tr key={r}>
                {row.map((col, c) => <td className="uiTableCell" key={c}>{this.cell(col)}</td>)}
            </tr>
        )
    }

    render() {
        let rows = [];

        for (let r = 0; r < this.props.numRows; r++) {
            let row = this.props.getRow(r)
            if (row)
                rows.push(row)
        }

        let cls = util.className(
            'uiTable',
            this.props.fixedLeftColumn ? 'uiTableWithFixedLeftColumn' : '',
            this.props.fixedRightColumn ? 'uiTableWithFixedRightColumn' : '',
        )

        return <div className={cls}>
            <table>
                {this.props.headers && <thead>{this.renderRows(this.props.headers)}</thead>}
                <tbody>{this.renderRows(rows)}</tbody>
                {this.props.footers && <tfoot>{this.renderRows(this.props.footers)}</tfoot>}
            </table>
        </div>
    }
}
