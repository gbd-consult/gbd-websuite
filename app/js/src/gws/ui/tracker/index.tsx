import * as React from 'react';
import * as util from '../util';


interface TrackerProps {
    xValueMin?: number;
    xValueMax?: number;
    yValueMin?: number;
    yValueMax?: number;

    xValue?: number;
    yValue?: number;

    noHandle?: boolean;
    noMove?: boolean;
    idleInterval?: number;

    whenPressed?: () => void;
    whenReleased?: () => void;
    whenChanged: (x: number, y: number) => void;
}

export class Tracker extends React.PureComponent<TrackerProps> {
    rootRef: React.RefObject<any>;
    handleRef: React.RefObject<any>;
    handlers: any;

    idleTimer: any;

    constructor(props) {
        super(props);

        this.rootRef = React.createRef();
        this.handleRef = React.createRef();

        this.handlers = {
            mousemove: this.onMouseMove.bind(this),
            mouseup: this.onMouseUp.bind(this),
            touchmove: this.onTouchMove.bind(this),
            touchend: this.onTouchEnd.bind(this),
            touchcancel: this.onTouchCancel.bind(this)
        }
    }

    render() {
        return <button
            className="uiTracker"
            tabIndex={-1}
            ref={this.rootRef}
            onMouseDown={e => this.onMouseDown(e)}
            onTouchStart={e => this.onTouchStart(e)}
        >
            {!this.props.noHandle && <div className="uiTrackerHandle" ref={this.handleRef}/>}
        </button>;
    }

    get dimensions() {
        let r = this.rootRef.current,
            h = this.handleRef.current,
            xmin = this.props.xValueMin || 0,
            xmax = this.props.xValueMax || 0,
            ymin = this.props.yValueMin || 0,
            ymax = this.props.yValueMax || 0;

        let s = {
            xMin: 0,
            xMax: 0,
            yMin: 0,
            yMax: 0,
            dx: 0,
            dy: 0,
        };

        if (xmin === xmax) {
            s.xMin = s.xMax = r.offsetWidth >> 1;
        } else if (h) {
            s.xMin = h.offsetWidth >> 1;
            s.xMax = r.offsetWidth - s.xMin;
        } else {
            s.xMin = 0;
            s.xMax = r.offsetWidth;
        }

        if (ymin === ymax) {
            s.yMin = s.yMax = r.offsetHeight >> 1;
        } else if (h) {
            s.yMin = h.offsetHeight >> 1;
            s.yMax = r.offsetHeight - s.yMin;
        } else {
            s.yMin = 0;
            s.yMax = r.offsetHeight;
        }

        s.dx = h ? (h.offsetWidth >> 1) : 0;
        s.dy = h ? (h.offsetHeight >> 1) : 0;

        return s;
    }

    componentDidMount() {
        this.updatePositionFromValues(this.props.xValue || 0, this.props.yValue || 0);

    }

    componentDidUpdate() {
        this.updatePositionFromValues(this.props.xValue || 0, this.props.yValue || 0);
    }

    componentWillUnmount() {
        this.clearIdle();
        this.trackStop();
    }

    //

    protected onTouchStart(e: React.TouchEvent<any>) {
        this.clearIdle();
        if (e.touches.length === 1) {
            this.trackStart(e.touches[0].clientX, e.touches[0].clientY);
            e.preventDefault();
            e.stopPropagation();
        }
    }

    protected onMouseDown(e: React.MouseEvent<any>) {
        this.clearIdle();
        this.trackStart(e.clientX, e.clientY);
        e.preventDefault();
        e.stopPropagation();
    }

    protected onMouseMove(e: MouseEvent) {
        this.clearIdle();
        if (this.props.noMove)
            return;
        this.trackUpdate(e.clientX, e.clientY);
        e.preventDefault();
    }

    protected onMouseUp(e: MouseEvent) {
        this.clearIdle();
        this.trackStop();
        e.preventDefault();
    }

    protected onTouchMove(e: TouchEvent) {
        this.clearIdle();
        if (this.props.noMove)
            return;
        if (e.touches.length === 1) {
            this.trackUpdate(e.touches[0].clientX, e.touches[0].clientY);
            e.preventDefault();
        }
    }

    protected onTouchEnd(e: TouchEvent) {
        this.clearIdle();
        this.trackStop();
        e.preventDefault();
    }

    protected onTouchCancel(e: TouchEvent) {
        this.clearIdle();
        this.trackStop();
        e.preventDefault();
    }

    //

    protected trackStart(x, y) {
        if (this.rootRef.current)
            this.rootRef.current.focus();

        if (this.props.whenPressed)
            this.props.whenPressed();


        let [vx, vy] = this.getValues(x, y);
        this.props.whenChanged(vx, vy);

        if (this.props.idleInterval) {
            // init interval must be longer to handle a simple click properly
            this.trackIdle(vx, vy, this.props.idleInterval * 3);
        }

        let doc = this.rootRef.current.ownerDocument;
        Object.keys(this.handlers).forEach(e => doc.addEventListener(e, this.handlers[e]));
    }

    protected trackStop() {
        let doc = this.rootRef.current.ownerDocument;
        Object.keys(this.handlers).forEach(e => doc.removeEventListener(e, this.handlers[e]));

        if (this.props.whenReleased)
            this.props.whenReleased();
    }

    protected trackUpdate(x, y) {
        let [vx, vy] = this.getValues(x, y);
        this.props.whenChanged(vx, vy);
        if (this.props.idleInterval)
            this.trackIdle(vx, vy, this.props.idleInterval);
    }

    protected trackIdle(vx, vy, interval) {
        this.idleTimer = setTimeout(() => this.callIdle(vx, vy), interval);
    }

    protected callIdle(vx, vy) {
        this.props.whenChanged(vx, vy);
        this.trackIdle(vx, vy, this.props.idleInterval);
    }

    protected clearIdle() {
        clearTimeout(this.idleTimer);
    }

    protected updatePositionFromValues(vx, vy) {
        let h = this.handleRef.current;

        if (h) {
            let d = this.dimensions;

            let
                px = util.translate(vx, this.props.xValueMin || 0, this.props.xValueMax || 0, d.xMin, d.xMax),
                py = util.translate(vy, this.props.yValueMin || 0, this.props.yValueMax || 0, d.yMin, d.yMax);

            h.style.left = (px - d.dx) + 'px';
            h.style.top = (py - d.dy) + 'px';
        }
    }

    protected getValues(x, y) {
        let d = this.dimensions;

        let b = this.rootRef.current,
            r = b.getBoundingClientRect(),
            px = util.constrain(x - r.left, d.xMin, d.xMax),
            py = util.constrain(y - r.top, d.yMin, d.yMax);

        return [
            util.translate(px, d.xMin, d.xMax, this.props.xValueMin || 0, this.props.xValueMax || 0),
            util.translate(py, d.yMin, d.yMax, this.props.yValueMin || 0, this.props.yValueMax || 0)
        ];
    }
}

