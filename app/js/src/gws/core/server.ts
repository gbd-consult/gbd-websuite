import axios, {Canceler, ResponseType} from 'axios';
import * as msgpack from '@ygoe/msgpack';
import * as api from './api';

function errorResponse(err) {
    console.log('AXIOS ERROR', err);
    return {
        error: {
            status: err.response ? err.response.status : -1,
            info: String(err)
        }
    }
}

enum QueueState {
    OPEN = 1,
    RUNNING = 2,
    DONE = 3
}

class QueuedRequest {
    layerUid: string;
    url: string;
    responseType: ResponseType;
    state: QueueState;
    server: Server;
    cancel: Canceler;
    resolve: any;
    promise: Promise<any>;

    constructor(server: Server, layerUid, url, responseType) {
        this.server = server;
        this.layerUid = layerUid;
        this.url = url;
        this.responseType = responseType;
        this.state = QueueState.OPEN;

        this.promise = new Promise<any>(resolve => this.resolve = resolve);
    }

    abort() {
        console.log('ABORT', this.layerUid)
        if (this.cancel) {
            this.cancel();
        }
        this.end(null);
    }

    end(data) {
        this.state = QueueState.DONE;
        if (this.resolve) {
            this.resolve(data);
        }
        this.resolve = null;
    }

    start() {
        console.log('START', this.layerUid);

        this.state = QueueState.RUNNING;

        axios.get(this.url, {
            cancelToken: new axios.CancelToken((c) => this.cancel = c),
            responseType: this.responseType
        })
            .then(res => {
                console.log('OK', this.layerUid);
                this.end(res.data)
            })
            .catch(err => {
                console.log('ERR', this.layerUid, err);
                this.end(null)
            });
    }

}

const MSGPACK_MIME = 'application/msgpack';

export class Server extends api.BaseServer {
    url: string;
    whenChanged: () => void;
    commandCount = 0;
    queue: Array<QueuedRequest> = [];
    queueSize = 3;

    protected app;

    constructor(app, url) {
        super();
        this.url = url;
        this.whenChanged = () => null;
        this.app = app;
        window.setInterval(() => this.pollQueue(), 500);
    }

    get requestCount() {
        return this.commandCount + this.queue.length;
    }

    async invoke(cmd, request, options) {
        request = request || {};
        request.projectUid = request.projectUid || this.app.project?.uid;
        request.localeUid = request.localeUid || this.app.localeUid;

        this.commandCount++;
        this.whenChanged();
        let res = await this._invoke2(cmd, request, options);
        this.commandCount--;
        this.whenChanged();
        return res;
    }

    async _invoke2(cmd, request, options) {
        let req: any = {
            url: this.url + '/' + cmd,
            method: 'POST',
            data: request,
            withCredentials: true
        };

        let binaryRequest = false;
        let binaryResponse = false;

        if (options) {
            binaryRequest = !!options.binaryRequest;
            binaryResponse = !!options.binaryResponse;
            if (options.binary)
                binaryRequest = binaryResponse = true;
        }

        req.headers = {}

        if (binaryRequest) {
            req.data = msgpack.serialize(req.data);
            req.headers['content-type'] = MSGPACK_MIME
        }
        if (binaryResponse) {
            req.headers['accept'] = MSGPACK_MIME;
            req.responseType = 'arraybuffer';
        }

        try {
            let res = await axios.request(req),
                data = res.data;
            if (res.headers['content-type'].includes(MSGPACK_MIME))
                data = msgpack.deserialize(new Uint8Array(data));
            return data;
        } catch (err) {
            if (err && err.response && err.response.data)
                return err.response.data;
            return errorResponse(err);
        }
    }

    queueLoad(layerUid, url, responseType) {
        this.dequeueLoad(layerUid);

        if (!url)
            return;

        let req = new QueuedRequest(this, layerUid, url, responseType);
        this.queue.push(req);

        console.log('QUEUE', layerUid);

        return req.promise;
    }

    dequeueLoad(layerUid) {
        this.queue = this.queue.filter(r => r.state !== QueueState.DONE);

        this.queue.forEach(r => {
            if (r.layerUid === layerUid)
                r.abort();
        });
    }

    prevLen = 0;

    pollQueue() {
        if (!this.queue.length && !this.prevLen)
            return;

        this.queue = this.queue.filter(r => r.state !== QueueState.DONE);
        this.prevLen = this.queue.length;

        if (this.queue.length === 0) {
            console.log('EMPTY');
            this.whenChanged();
            return;
        }

        let running = this.queue.filter(r => r.state === QueueState.RUNNING).length;
        let open = this.queue.filter(r => r.state === QueueState.OPEN);

        //console.log('POLL', running, '/', this.queue.length, this.queue.map(r => r.layerUid))

        if (running < this.queueSize && open.length) {
            open[0].start();
            this.whenChanged();
        }
    }
}

