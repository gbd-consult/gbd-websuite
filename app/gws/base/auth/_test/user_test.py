import gws
import gws.test.util as u
import gws.base.auth


@u.fixture(scope='module')
def root():
    cfg = '''
        auth {
            providers+ {
                uid "AUTH_1"
                type 'mockAuthProvider1'
                allowedMethods ['mockAuthMethod1']
            }
            methods+ { type 'mockAuthMethod2' }
        }
    '''

    yield u.gws_root(cfg)


##


def test_from_record(root: gws.Root):
    prov = root.app.authMgr.get_provider('AUTH_1')
    rec = dict(
        localUid='a',
        displayName='b',
        mfauid='MFA',
        email='bob',
        other1='x1',
        other2='x2',
    )
    usr = gws.base.auth.user.from_record(prov, rec)

    assert usr.uid == gws.u.join_uid(prov.uid, rec['localUid'])
    assert usr.displayName == 'b'
    assert usr.mfaUid == 'MFA'
    assert usr.email == 'bob'
    assert usr.data == dict(other1='x1', other2='x2')
