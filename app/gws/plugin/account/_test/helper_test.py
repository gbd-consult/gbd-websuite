import gws
import gws.test.util as u
import gws.plugin.account.core as core


@u.fixture(scope='module')
def root():
    u.pg.create('accounts', {
        'id': 'int primary key',
        'email': 'text',
        'status': 'int',
        'password': 'text',
        'mfauid': 'text',
        'mfasecret': 'text',
        'tc': 'text',
        'tctime': 'int',
        'tccategory': 'text',
    })
    cfg = '''
        helpers+ {
            type "account"
            onboardingUrl "http://example.com/onboarding"
            passwordCreateSql "md5({password})"
            passwordVerifySql "md5({password})"
            adminModel {
                type "postgres"
                tableName "accounts"
            }
        }
    '''
    yield u.gws_root(cfg)


@u.fixture(autouse=True)
def account():
    u.pg.clear('accounts')
    u.pg.insert('accounts', [{'id': 1, 'email': 'user_1@example.com', 'status': core.Status.new}])


def _helper(root):
    return root.app.helper('account')


def test_tc_can_be_used_once(root):
    h = _helper(root)
    acc = h.get_account_by_id(1)
    tc = h.generate_tc(acc, core.Category.onboarding)

    assert h.get_account_by_tc(tc, core.Category.onboarding)['id'] == 1
    assert h.get_account_by_tc(tc, core.Category.onboarding) is None


def test_tc_is_invalidated_for_wrong_category(root):
    h = _helper(root)
    acc = h.get_account_by_id(1)
    tc = h.generate_tc(acc, core.Category.onboarding)

    assert h.get_account_by_tc(tc, core.Category.resetPassword) is None
    assert h.get_account_by_tc(tc, core.Category.onboarding) is None


def test_set_password(root):
    h = _helper(root)
    acc = h.get_account_by_id(1)
    h.set_password(acc, 'password_1')
    h.set_status(acc, core.Status.active)
    found = h.get_account_by_credentials(gws.Data(username='user_1@example.com', password='password_1'))
    assert found['id'] == 1
