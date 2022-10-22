"""Test runner.

Assumes `app/test.py` has been invoked on the host and everything is set up.
"""

import sys
import gws.lib.test.container_runner

if __name__ == '__main__':
    sys.exit(gws.lib.test.container_runner.main(sys.argv))
