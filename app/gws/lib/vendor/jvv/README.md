# jvv, the json viewer

jvv is a simple browser-based JSON viewer. Its main advantage is that the source JSON remains correctly copyable.

Usage:

```html
<link rel="stylesheet" href="./jvv.css">
<script src="./jvv.js"></script>

<script>
    const myJson = {...}

    const myContainer = document.getElementById(...) 
    
    const options = {
        navBar: true,
        arrayHints: true,
        depth: 3,
    } 
    
    JsonViewer.create(
        myJson, 
        myContainer,
        options
    )
    
</script>
```
