"""Storage of the deltas sent by QField, with their status."""

from typing import Optional

import gws
import gws.lib.jsonx
import gws.lib.datetimex as dtx

from . import core, api, patcher


PENDING_TIMEOUT = 300
"""Time in seconds after which a pending delta is considered failed."""

STORAGE_DDL = """
    CREATE TABLE IF NOT EXISTS deltas (
        uid TEXT PRIMARY KEY,
        created_by TEXT NOT NULL,
        id TEXT NOT NULL,
        client_id TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        status TEXT NOT NULL,
        output TEXT,
        content TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS deltas_created_at ON deltas (created_at);
    CREATE INDEX IF NOT EXISTS deltas_created_by_created_at ON deltas (created_by, created_at);
    CREATE TABLE IF NOT EXISTS payloads (
        created_by TEXT NOT NULL,
        payload_id TEXT NOT NULL,
        delta_uid TEXT NOT NULL,
        PRIMARY KEY (created_by, payload_id, delta_uid)
    );
    CREATE INDEX IF NOT EXISTS payloads_delta_uid ON payloads (delta_uid);
"""


class Record(gws.Data):
    """A delta as stored on the server, with its status."""

    uid: str
    """Record uid, a hash of the user and the delta content."""
    id: str
    """Delta id (the ``uuid`` of the delta)."""
    created_by: str
    """Login name of the user who sent the delta."""
    client_id: str
    """Id of the local export on the device."""
    created_at: int
    """Creation time as a timestamp."""
    updated_at: int
    """Last update time as a timestamp."""
    status: api.DeltaStatusType
    """Delta status."""
    output: str
    """Error message, if the delta could not be applied."""
    content: api.Delta
    """Delta as sent by QField."""


def parse_payload(text: str) -> api.DeltasPayload:
    """Parse the content of a delta file, raise ``gws.BadRequestError`` if it is invalid."""
    try:
        js = gws.lib.jsonx.from_string(text)
        return api.DeltasPayload(
            deltas=[_from_dict(d) for d in js['deltas']],
            files=js.get('files', []),
            id=js['id'],
            project=js['project'],
            version=js['version'],
        )
    except Exception as exc:
        raise gws.BadRequestError(f'invalid delta file content: {exc}')


def new_record(uid: str, d: api.Delta, user_name: str) -> Record:
    """Create a pending delta record for a delta."""
    now = dtx.to_timestamp()
    return Record(
        uid=uid,
        id=d.uuid,
        created_by=user_name,
        client_id=d.clientId or '',
        created_at=now,
        updated_at=now,
        status=api.DeltaStatusType.pending,
        output='',
        content=d,
    )


def store_all(db: gws.lib.sqlitex.Object, drs: list[Record]):
    """Store delta records, replacing the stored ones with the same uid."""
    for dr in drs:
        db.insert('deltas', _row_from_record(dr), on_conflict='update')


def link_payload(db: gws.lib.sqlitex.Object, payload_id: str, delta_uids: list[str], user_name: str):
    """Link a payload to its delta records, replacing the links stored for the same payload and user."""
    db.execute(
        'DELETE FROM payloads WHERE created_by=:user_name AND payload_id=:payload_id',
        user_name=user_name,
        payload_id=payload_id,
    )
    for uid in delta_uids:
        db.insert('payloads', {'created_by': user_name, 'payload_id': payload_id, 'delta_uid': uid}, on_conflict='ignore')


def extract_changes(drs: list[Record]) -> list[patcher.Change]:
    """Convert delta records to patcher changes."""
    changes = []
    for dr in drs:
        d = dr.content
        changes.append(
            patcher.Change(
                uid=d.uuid,
                type=d.method,
                layerUid=d.localLayerId,
                newAtts=(d.new.attributes or {}) if d.new else {},
                oldAtts=(d.old.attributes or {}) if d.old else {},
                wkt=(d.new.geometry or '') if d.new else '',
            )
        )
    return changes


def set_status_for_all(db: gws.lib.sqlitex.Object, drs: list[Record], status: api.DeltaStatusType, output: str = ''):
    """Set the status and output of delta records, in the objects and in the database."""
    now = dtx.to_timestamp()

    for dr in drs:
        dr.status = status
        dr.output = output
        dr.updated_at = now

    store_all(db, drs)


def get_one(db: gws.lib.sqlitex.Object, uid: str, user_name: str) -> Optional[Record]:
    """Load a delta record of a user, ``None`` if not found."""
    rs = db.select(
        'SELECT * FROM deltas WHERE uid=:uid AND created_by=:user_name',
        uid=uid,
        user_name=user_name,
    )
    return _record_from_row(rs[0]) if rs else None


def get_for_payload(db: gws.lib.sqlitex.Object, payload_id: str, user_name: str) -> list[Record]:
    """Load the delta records of a payload sent by a user, empty if not found."""
    rs = db.select(
        """
            SELECT deltas.* FROM payloads JOIN deltas ON deltas.uid = payloads.delta_uid
            WHERE payloads.created_by=:user_name AND payloads.payload_id=:payload_id
            ORDER BY payloads.rowid
        """,
        user_name=user_name,
        payload_id=payload_id,
    )
    return [_record_from_row(r) for r in rs]


def get_history(db: gws.lib.sqlitex.Object, user_name: str, limit: int, offset: int) -> list[tuple[Record, str]]:
    """Load delta records of a user, newest first, each with the id of the last payload it was sent with."""
    rs = db.select(
        """
            SELECT
                deltas.*,
                COALESCE((
                    SELECT payload_id FROM payloads
                    WHERE payloads.delta_uid = deltas.uid
                    ORDER BY payloads.rowid DESC LIMIT 1
                ), '') AS payload_id
            FROM deltas
            WHERE created_by=:user_name
            ORDER BY created_at DESC, rowid DESC
            LIMIT :limit OFFSET :offset
        """,
        user_name=user_name,
        limit=limit,
        offset=offset,
    )
    return [(_record_from_row(r), r['payload_id']) for r in rs]


def status(dr: Record, payload_id: str) -> core.DeltaStatus:
    """Convert a delta record to the status returned for a payload, reporting a stale pending delta as failed."""
    now = dtx.now().timestamp()

    status = dr.status
    output = dr.output

    # delta record has been pending for longer than ``PENDING_TIMEOUT``
    if dr.status == api.DeltaStatusType.pending and now - dr.updated_at > PENDING_TIMEOUT:
        status = api.DeltaStatusType.error
        output = 'timed out'

    return core.DeltaStatus(
        id=dr.id,
        deltafile_id=payload_id,
        created_by=dr.created_by,
        created_at=dtx.to_iso_string(dtx.from_timestamp(dr.created_at)),
        updated_at=dtx.to_iso_string(dtx.from_timestamp(dr.updated_at)),
        status=status,
        output=output,
        content=dr.content,
    )


def cleanup(db: gws.lib.sqlitex.Object, life_time: int):
    """Remove delta records older than ``life_time`` seconds, and the payload links to them."""
    db.execute('DELETE FROM deltas WHERE created_at < :t', t=dtx.to_timestamp() - life_time)
    db.execute('DELETE FROM payloads WHERE delta_uid NOT IN (SELECT uid FROM deltas)')


def _from_dict(d: dict) -> api.Delta:
    """Convert a delta from the payload to an API delta."""
    for key in ('uuid', 'method', 'localLayerId'):
        if not d.get(key):
            raise ValueError(f'delta: missing {key!r}')
    new = d.get('new')
    old = d.get('old')
    return api.Delta(
        d,
        new=api.DeltaFeature(new) if new else None,
        old=api.DeltaFeature(old) if old else None,
    )


def _row_from_record(dr: Record) -> dict:
    """Convert a delta record to a database row."""
    return dict(
        uid=dr.uid,
        created_by=dr.created_by,
        id=dr.id,
        client_id=dr.client_id,
        created_at=dr.created_at,
        updated_at=dr.updated_at,
        status=dr.status,
        output=dr.output,
        content=gws.lib.jsonx.to_string(dr.content),
    )


def _record_from_row(rec: dict) -> Record:
    """Convert a database row to a delta record."""
    return Record(
        uid=rec['uid'],
        id=rec['id'],
        created_by=rec['created_by'],
        client_id=rec['client_id'] or '',
        created_at=rec['created_at'],
        updated_at=rec['updated_at'],
        status=rec['status'],
        output=rec['output'] or '',
        content=_from_dict(gws.lib.jsonx.from_string(rec['content'])),
    )
