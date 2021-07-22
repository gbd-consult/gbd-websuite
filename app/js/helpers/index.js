// npm scripts entry point

const builder = require('./builder')

function main(argv) {
    let args = {}
    let key = 'command'

    for (let a of argv.slice(2)) {
        if (a.startsWith('--')) {
            key = a.slice(2)
            args[key] = true
        } else {
            args[key] = a
        }
    }

    let b = new builder.Builder()
    b.run(args)
}

main(process.argv)
