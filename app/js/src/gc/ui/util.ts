export function className(...classNames) {
    return classNames
        .filter(Boolean)
        .map(s => String(s).trim())
        .filter(Boolean)
        .join(' ');

}

export function number(x, def: number = 0): number {
    return (typeof x === 'number') ? x : (Number(x) || def);
}

export function nextTick(fn) {
    setTimeout(fn, 0);
}

export function empty(v): boolean {
    return typeof v === 'undefined'
        || v === null
        || v === ''
        || (Array.isArray(v) && v.length < 1)
        || (typeof v === 'object' && Object.keys(v).length < 1);
}


export function constrain(n: number, min: number, max: number): number {
    if (Number.isNaN(n))
        return min;
    return Math.max(min, Math.min(max, n));
}

export function translate(x, minSrc: number, maxSrc: number, minDst: number, maxDst: number): number {
    if (minSrc === maxSrc)
        return minDst;
    x = constrain(x, minSrc, maxSrc);
    return minDst + (minDst - maxDst) * (x - minSrc) / (minSrc - maxSrc)
}

export function align(n: number, step: number, delta: number = 0): number {
    if (!step)
        return n;

    let s;

    if (delta < 0) {
        s = Math.ceil(n / step)
    } else if (delta > 0) {
        s = Math.floor(n / step)
    } else {
        s = Math.round(n / step)
    }
    return (s + delta) * step;
}

export function sign(n: number): number {
    if (n < 0) return -1;
    if (n > 0) return +1;
    return 0;
}

export function parseNumber(s): number {
    if (empty(s))
        return NaN;
    return Number(s);
}

// matches the server format (see gc.tool.intl)
export interface Locale {
    dateFormatLong: string;
    dateFormatMedium: string;
    dateFormatShort: string;
    dateUnits: string;
    dayNamesLong: Array<string>;
    dayNamesNarrow: Array<string>;
    dayNamesShort: Array<string>;
    firstWeekDay: number;
    language: string;
    languageName: string;
    monthNamesLong: Array<string>;
    monthNamesNarrow: Array<string>;
    monthNamesShort: Array<string>;
    numberDecimal: string;
    numberGroup: string;
}


export function formatNumber(n: number, fmt?: Locale): string {

    if (empty(n) || Number.isNaN(n))
        return '';
    // @TODO
    let ns = String(n);
    if (fmt) {
        if (fmt.numberDecimal) {
            ns = ns.replace('.', fmt.numberDecimal);
        }
    }
    return ns;
}

export interface DMY {
    d: number;
    m: number;
    y: number;
};

export function formatDate(dmy: DMY, fmt: string, lo: Locale): string {
    // support a small subset of
    // https://www.unicode.org/reports/tr35/tr35-dates.html#Date_Field_Symbol_Table

    let lz = n => (n < 10 ? '0' : '') + String(n);

    return fmt.replace(/yyyy|yy|MMMM|MMM|MM|M|dd|d/g, p => {
        switch (p) {
            case 'yyyy':
                return String(dmy.y);
            case 'yy':
                return lz(dmy.y % 100);
            case 'MMMM':
                return lo.monthNamesLong[dmy.m - 1]
            case 'MMM':
                return lo.monthNamesShort[dmy.m - 1]
            case 'MM':
                return lz(dmy.m);
            case 'M':
                return String(dmy.m);
            case 'dd':
                return lz(dmy.d);
            case 'd':
                return String(dmy.d);
        }
    });
}

export function iso2dmy(val): DMY {
    let s = String(val || '').trim().match(/^(\d+)-(\d+)-(\d+)/);
    if (!s) {
        return null;
    }

    let y = Number(s[1]);
    let m = Number(s[2]);
    let d = Number(s[3]);

    return {d, m, y};
}

export function date2dmy(date: Date): DMY {
    return {
        y: date.getFullYear(),
        m: date.getMonth() + 1,
        d: date.getDate(),
    }
}

export function dmy2iso(dmy: DMY): string {
    let s = String(dmy.y) + '-';
    if (dmy.m < 10)
        s += '0';
    s += String(dmy.m) + '-';
    if (dmy.d < 10)
        s += '0';
    s += String(dmy.d);
    return s;
}

export function date2iso(date?: Date): string {
    return dmy2iso(date2dmy(date || new Date()));
}


export function range(a: number, b?: number, step?: number): Array<number> {


    switch (arguments.length) {
        case 0:
            return [];
        case 1:
            b = Number(a);
            a = 0;
            step = 1;
            break;
        case 2:
            a = Number(a);
            b = Number(b);
            step = a < b ? +1 : -1;
            break;
        case 3:
            a = Number(a);
            b = Number(b);
            step = Number(step);
            break;
        default:
            return [];
    }

    if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(step))
        return [];

    if (a < b) {
        if (step <= 0)
            return [];

        let r = [];

        while (a < b) {
            r.push(a);
            a += step;
        }
        return r;
    }

    if (a > b) {
        if (step >= 0)
            return [];

        let r = [];

        while (a > b) {
            r.push(a);
            a += step;
        }

        return r;
    }

    return [];
}
