document.getElementById('trendform').addEventListener('submit', async function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    try {
        const response = await fetch('/trend/', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '';

        if (data.success) {
            resultDiv.innerHTML = `
                <h3>Predicted Trend for ${data.state}</h3>
                
                <img src="${data.plot_path}" alt="Trend Plot" style="max-width: 100%;">
            `;
        } else {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        }
    } catch (err) {
        console.error('Unexpected error:', err);
        document.getElementById('result').innerHTML =
            `<p style="color: red;">Unexpected error: ${err.message}</p>`;
    }
});
