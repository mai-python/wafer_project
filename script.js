function sendCommand(action) {
    fetch("/send_command", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action: action })
    });
}

setInterval(() => {
    fetch("/status")
        .then(res => res.json())
        .then(data => {
            document.getElementById("wafer_type").innerText = data.wafer_type;
            document.getElementById("dx").innerText = data.dx.toFixed(2);
            document.getElementById("dy").innerText = data.dy.toFixed(2);
            document.getElementById("eta").innerText = data.time.toFixed(2);

            document.getElementById("current_status").innerText = capitalize(data.current);
            document.getElementById("recognition_status").innerText = capitalize(data.recognition);
            document.getElementById("motor_status").innerText = capitalize(data.motor);
            document.getElementById("error_status").innerText = data.error;

            updateBadge("current_status", data.current);
            updateBadge("recognition_status", data.recognition);
            updateBadge("motor_status", data.motor);
            updateBadge("error_status", data.error);
        });
}, 1000);

setInterval(() => {
    fetch("/logs")
        .then(res => res.json())
        .then(data => {
            const panel = document.getElementById("log_panel");
            panel.innerHTML = "";
            data.logs.slice(-10).forEach(line => {
                const p = document.createElement("p");
                p.innerText = line;
                panel.appendChild(p);
            });
        });
}, 2000);

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function updateBadge(id, state) {
    const el = document.getElementById(id);
    el.className = "badge " + state;
}

