// This script loads the JSON variable from data.js and displays it in the HTML

window.onload = function () {
    // Access the JSON data from data.js
    const jsonData = data // Assuming 'data' is the variable in data.js
    const contentDiv = document.getElementById("content")

    // Display the JSON data
    contentDiv.innerHTML = `<pre>${JSON.stringify(jsonData, null, 2)}</pre>`
}
