"""Run the tests

Assumes /test.py has been invoked on the host and everything is set up.
"""

import sys
import gws.lib.test

sys.exit(gws.lib.test.main(sys.argv[1:]) or 0)
