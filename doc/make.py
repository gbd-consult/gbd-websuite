"""Doc generator CLI tool"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../app/gws/lib/vendor'))

import options
import dog

if __name__ == '__main__':
    sys.exit(dog.run(sys.argv, options.OPTIONS))
