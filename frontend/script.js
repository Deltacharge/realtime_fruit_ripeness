const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

function capture() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("file", blob, "capture.jpg");

        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            document.getElementById("result").innerHTML =
                `${data.label.toUpperCase()} (${(data.confidence*100).toFixed(1)}%)`;
        });
    }, "image/jpeg");
}


imageInput.addEventListener("change", () => {
    preview.innerHTML = "";
    const file = imageInput.files[0];
    if (!file) return;

    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    preview.appendChild(img);
});

function predict() {
    const file = imageInput.files[0];
    if (!file) {
        alert("Please upload an image first!");
        return;
    }

    resultDiv.innerHTML = "⏳ Predicting...";

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = "❌ Error: " + data.error;
        } else {
            const confidence = (data.confidence * 100).toFixed(2);
            resultDiv.innerHTML = `✅ ${data.label.toUpperCase()} <br> Confidence: ${confidence}%`;
        }
    })
    .catch(err => {
        resultDiv.innerHTML = "❌ Server error";
        console.error(err);
    });
}
