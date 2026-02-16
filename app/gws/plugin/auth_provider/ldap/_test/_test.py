import yaml
import os

import gws
import gws.test.util as u

"""
Tests for LDAP auth provider using test.ldif structure.
"""

@u.fixture(scope='module')
def root():
    src = os.path.dirname(__file__)
    with open(f'{src}/certs.yaml') as fp:
        certs = yaml.safe_load(fp)
        for name, content in certs.items():
            with open(f'/tmp/{name}', 'w') as fp:
                fp.write(content)
        os.chmod(f'/tmp/ldap.key', 0o600)

    cfg = """
        auth {
            methods+ { type 'basic' }
            providers+ { 
                uid "ldap_nor"
                type 'ldap' 
                url 'ldap://cldap:389/dc=example,dc=com?uid'
                bindDN 'cn=admin,dc=example,dc=com'
                bindPassword 'gispass'
                displayNameFormat '{cn} ({mail})'
                users+ {
                    roles ['manager']
                    matches '(title=manager)'
                }
                users+ {
                    roles ['sales']
                    matches '(departmentNumber=sales)'
                }
                users+ {
                    roles ['group_one']
                    memberOf 'cn=groupone'
                }
                users+ {
                    roles ['group_two']
                    memberOf 'cn=grouptwo'
                }
            }
            providers+ { 
                uid "ldap_ssl"
                type 'ldap' 
                url 'ldaps://cldap:636/dc=example,dc=com?uid'
                bindDN 'cn=admin,dc=example,dc=com'
                bindPassword 'gispass'
                ssl {
                    ca  "/tmp/ca.crt"
                    crt "/tmp/ldap.crt"
                    key "/tmp/ldap.key"
                }
            }
        }
    """

    yield u.gws_root(cfg)


##


def test_authenticate_valid_user(root: gws.Root):
    """Test authentication with valid credentials"""
    am = root.app.authMgr
    prv = am.providers[0]
    usr = prv.authenticate(am.methods[0], gws.Data(username='b', password='bpass'))
    assert usr is not None
    assert usr.loginName == 'b'
    assert usr.localUid == 'b'


def test_authenticate_wrong_password(root: gws.Root):
    """Test authentication with wrong password raises ForbiddenError"""
    am = root.app.authMgr
    prv = am.providers[0]
    try:
        prv.authenticate(am.methods[0], gws.Data(username='a', password='wrongpass'))
        assert False, "Should have raised ForbiddenError"
    except gws.ForbiddenError:
        pass


def test_authenticate_nonexistent_user(root: gws.Root):
    """Test authentication with non-existent user returns None"""
    am = root.app.authMgr
    prv = am.providers[0]
    usr = prv.authenticate(am.methods[0], gws.Data(username='nonexistent', password='pass'))
    assert usr is None


def test_authenticate_missing_credentials(root: gws.Root):
    """Test authentication with missing username or password"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    # Missing username
    usr = prv.authenticate(am.methods[0], gws.Data(password='apass'))
    assert usr is None
    
    # Missing password
    usr = prv.authenticate(am.methods[0], gws.Data(username='a'))
    assert usr is None
    
    # Missing both
    usr = prv.authenticate(am.methods[0], gws.Data())
    assert usr is None


def test_get_user_existing(root: gws.Root):
    """Test get_user with existing user"""
    am = root.app.authMgr
    prv = am.providers[0]
    usr = prv.get_user('c')
    assert usr is not None
    assert usr.loginName == 'c'
    assert usr.localUid == 'c'


def test_get_user_nonexistent(root: gws.Root):
    """Test get_user with non-existent user returns None"""
    am = root.app.authMgr
    prv = am.providers[0]
    usr = prv.get_user('nonexistent')
    assert usr is None


def test_roles_match_filter_manager(root: gws.Root):
    """Test role assignment based on LDAP attribute filter (title=manager)"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    # User 'a' has title=manager
    usr = prv.authenticate(am.methods[0], gws.Data(username='a', password='apass'))
    assert 'manager' in usr.roles
    
    # User 'd' has title=manager
    usr = prv.authenticate(am.methods[0], gws.Data(username='d', password='dpass'))
    assert 'manager' in usr.roles
    
    # User 'c' does not have title=manager
    usr = prv.authenticate(am.methods[0], gws.Data(username='c', password='cpass'))
    assert 'manager' not in usr.roles


def test_roles_match_filter_sales(root: gws.Root):
    """Test role assignment based on LDAP attribute filter (departmentNumber=Sales)"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    # User 'b' has departmentNumber=Sales
    usr = prv.authenticate(am.methods[0], gws.Data(username='b', password='bpass'))
    assert 'sales' in usr.roles
    
    # User 'e' has departmentNumber=Sales
    usr = prv.authenticate(am.methods[0], gws.Data(username='e', password='epass'))
    assert 'sales' in usr.roles
    
    # User 'a' does not have departmentNumber=Sales
    usr = prv.authenticate(am.methods[0], gws.Data(username='a', password='apass'))
    assert 'sales' not in usr.roles


def test_roles_memberof_groupone(root: gws.Root):
    """Test role assignment based on group membership (groupone)"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    # User 'a' is member of groupone
    usr = prv.authenticate(am.methods[0], gws.Data(username='a', password='apass'))
    assert 'group_one' in usr.roles
    
    # User 'b' is member of groupone
    usr = prv.authenticate(am.methods[0], gws.Data(username='b', password='bpass'))
    assert 'group_one' in usr.roles
    
    # User 'c' is NOT member of groupone
    usr = prv.authenticate(am.methods[0], gws.Data(username='c', password='cpass'))
    assert 'group_one' not in usr.roles


def test_roles_memberof_grouptwo(root: gws.Root):
    """Test role assignment based on group membership (grouptwo)"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    # User 'b' is member of grouptwo
    usr = prv.authenticate(am.methods[0], gws.Data(username='b', password='bpass'))
    assert 'group_two' in usr.roles
    
    # User 'c' is member of grouptwo
    usr = prv.authenticate(am.methods[0], gws.Data(username='c', password='cpass'))
    assert 'group_two' in usr.roles
    
    # User 'a' is NOT member of grouptwo
    usr = prv.authenticate(am.methods[0], gws.Data(username='a', password='apass'))
    assert 'group_two' not in usr.roles


def test_roles_multiple_assignments(root: gws.Root):
    """Test user with multiple role assignments"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    # User 'b' is member of groupone, grouptwo, and has departmentNumber=Sales
    usr = prv.authenticate(am.methods[0], gws.Data(username='b', password='bpass'))
    assert 'group_one' in usr.roles
    assert 'group_two' in usr.roles
    assert 'sales' in usr.roles
    assert 'manager' not in usr.roles


def test_display_name_format(root: gws.Root):
    """Test display name formatting"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    usr = prv.authenticate(am.methods[0], gws.Data(username='b', password='bpass'))
    assert usr.displayName == 'BName (bmail)'
    
    usr = prv.authenticate(am.methods[0], gws.Data(username='c', password='cpass'))
    assert usr.displayName == 'CName (cmail)'


def test_get_user_with_roles(root: gws.Root):
    """Test get_user returns user with correct roles"""
    am = root.app.authMgr
    prv = am.providers[0]
    
    # User 'a' has manager role and group_one
    usr = prv.get_user('a')
    assert usr is not None
    assert 'manager' in usr.roles
    assert 'group_one' in usr.roles
    
    # User 'd' has manager role only
    usr = prv.get_user('d')
    assert usr is not None
    assert 'manager' in usr.roles
    assert 'group_one' not in usr.roles
    assert 'group_two' not in usr.roles


def test_ssl_connection_authenticate(root: gws.Root):
    """Test authentication over SSL connection"""
    am = root.app.authMgr
    prv = am.providers[1]
    
    usr = prv.authenticate(am.methods[0], gws.Data(username='c', password='cpass'))
    assert usr is not None
    assert usr.loginName == 'c'


def test_ssl_connection_get_user(root: gws.Root):
    """Test get_user over SSL connection"""
    am = root.app.authMgr
    prv = am.providers[1]
    
    usr = prv.get_user('e')
    assert usr is not None
    assert usr.loginName == 'e'
