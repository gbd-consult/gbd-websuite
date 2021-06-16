import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

import {Row, Cell} from '../layout';


interface GridProps {
    className?: string;
}

export class Grid extends base.Pure<GridProps> {
    render() {
        let cls = util.className('uiGrid', this.props.className);
        return <div className={cls}>{this.props.children}</div>;
    }
}

const DEFAULT_UNIT = 4;
const DEFAULT_WIDTH = 40;
const DEFAULT_HEIGHT = 10;

interface TableProps {
    getRow: (r: number) => Array<React.ReactNode>;
    headers?: Array<Array<string | React.ReactNode>>;
    fixedColumns?: number;
    numRows?: number;
    selectedRow?: number;
    unit?: number;
    widths?: Array<number>;
}


// to fix "Type 'string' is not assignable to type 'FlexDirectionProperty'" and alikes

const _COLUMN: any = 'column';
const _ROW: any = 'row';
const _HIDDEN: any = 'hidden';
const _AUTO: any = 'auto';
const _100: any = '100%';


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

    onScroll(e) {
        if (e.target === this.bodyRef.current) {

            if (this.topRef.current) {
                this.topRef.current.querySelectorAll('.uiTableHeadScrollable').forEach(d =>
                    d.firstChild.style.left = -e.target.scrollLeft + 'px'
                );
            }

            if (this.leftRef.current) {
                this.leftRef.current.firstChild.style.top = -e.target.scrollTop + 'px';
            }
        }
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

    width(p, n) {
        if (n < p.widths.length)
            return p.widths[n];
        return (this.props.unit || DEFAULT_UNIT) * DEFAULT_WIDTH;
    }

    cell(p, n, val) {
        let w = this.width(p, n),
            q = w ? {width: w} : {flex: true};

        // @TODO flex width doesn't work with scrolled body, except for the last column
        // (scrollbar shifts the content to the left)

        if (typeof val === "string") {
            val = this.label(val);
        }

        return <Cell key={n} className="uiTableCell" {...q}>{val}</Cell>
    }

    label(s) {
        return <div className="uiTableStaticText">{s}</div>
    }

    body(p) {
        let sBox = {
            overflow: _AUTO,
            width: _100,
            height: _100,
        };

        let selected = this.props.selectedRow;

        return <div style={sBox} className="uiTableBody" ref={this.bodyRef} onScroll={e => this.onScroll(e)}>
            <Grid>
                {p.rows.map((row, r) =>
                    <div className="uiRow" key={r} ref={r === selected ? this.focusRef : null}>
                        {row.slice(p.fixedColumns).map((val, c) => this.cell(p, c + p.fixedColumns, val))}
                    </div>
                )}
            </Grid>
        </div>
    }


    leftCols(p) {
        let sBox = {
            display: 'flex',
            flexDirection: _COLUMN
        };

        let sDown = {
            flex: 1,
            overflow: _HIDDEN,
        };

        return <div style={sBox}>
            <div style={sDown} className="uiTableFixed" ref={this.leftRef}>
                <Grid>
                    {p.rows.map((row, r) =>
                        <Row key={r}>
                            {util.range(p.fixedColumns).map(c => this.cell(p, c, row[c]))}
                        </Row>
                    )}
                </Grid>
            </div>
        </div>
    }

    hbox(p) {
        if (p.fixedColumns === 0) {
            return this.body(p)
        }

        let sBox = {
            width: _100,
            height: _100,
            display: 'flex',
            flexDirection: _ROW,
            overflow: _HIDDEN,
        };

        let sLeft = {};

        let sRight = {
            flex: 1,
            overflow: _HIDDEN,
        };

        return <div style={sBox}>
            <div style={sLeft}>
                {this.leftCols(p)}
            </div>
            <div style={sRight}>
                {this.body(p)}
            </div>
        </div>
    }


    topRow(p, n) {
        let sBox = {
            width: _100,
            display: 'flex',
            flexDirection: _ROW,
            overflow: _HIDDEN,
        };

        let sLeft = {};

        let sRight = {
            flex: 1,
            overflow: _HIDDEN,
        };

        return <div style={sBox} key={n}>
            {p.fixedColumns > 0 && <div style={sLeft} className="uiTableHeadFixed">
                <Grid>
                    <Row>
                        {p.headers[n].slice(0, p.fixedColumns).map((val, c) => this.cell(p, c, val))}
                    </Row>
                </Grid>
            </div>}
            <div style={sRight} className="uiTableHeadScrollable">
                <Grid>
                    <Row>
                        {p.headers[n].slice(p.fixedColumns).map((val, c) => this.cell(p, c + p.fixedColumns, val))}
                    </Row>
                </Grid>
            </div>
        </div>
    }

    topRows(p) {
        return <div className="uiTableHead" ref={this.topRef}>
            {p.headers.map((_, n) => this.topRow(p, n))}
        </div>
    }

    vbox(p) {
        if (!p.headers) {
            return this.hbox(p)
        }

        let sBox = {
            display: 'flex',
            flexDirection: _COLUMN,
            overflow: _HIDDEN,
            width: _100,
            height: _100,
        };

        let sUp = {
            overflow: _HIDDEN,
        };

        let sDown = {
            flex: 1,
            overflowX: _AUTO,
            overflowY: _HIDDEN,
        };

        return <div style={sBox}>
            <div style={sUp}>
                {this.topRows(p)}
            </div>
            <div style={sDown}>
                {this.hbox(p)}
            </div>
        </div>
    }

    render() {
        let p = {
            fixedColumns: this.props.fixedColumns || 0,
            headers: this.props.headers,
            widths: this.props.widths || [],
            rows: util.range(this.props.numRows).map(r => this.props.getRow(r)).filter(Boolean),
        };

        return <div className='uiTable'>
            {this.vbox(p)}
        </div>;

    }
}
