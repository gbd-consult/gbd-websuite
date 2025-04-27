export function registerTags(tags) {
    Object.assign(registerTags.tags, tags)
}
registerTags.tags = {}

export function getRegisteredTags() {
    return registerTags.tags
}

