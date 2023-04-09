// npm scripts entry point

const builder = require('./builder')

function main(argv) {
    let args = {}
    let opt = null
    let n = 0

    for (let a of argv.slice(2)) {
        if (a.startsWith('-')) {
            opt = a.slice(1)
            args[opt] = true
        } else if (a.startsWith('--')) {
            opt = a.slice(2)
            args[opt] = true
        } else if (opt) {
            args[opt] = a
            opt = null
        } else {
            args[n] = a
            n += 1
        }
    }

    let b = new builder.Builder()
    b.run(args)
}

main(process.argv)
