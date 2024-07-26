let $ = (sel, parent) => (parent || document).querySelector(sel);

async function generateTotp() {
    let res = await window.fetch('?generate=1&secret=' + $('#secret_input').value);
    let js = await res.json();
    $('#totp_output').value = js.totp;
    $('#qr_code').src = js.image;
    $('#base32_secret').textContent = js.base32;
}

window.addEventListener('load', () => {
    generateTotp()
    $('#reload_button').addEventListener('click', generateTotp)
    $('#secret_input').addEventListener('input', generateTotp)
    setInterval(generateTotp, 10 * 1000)
});
