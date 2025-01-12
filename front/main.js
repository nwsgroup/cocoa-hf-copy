import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
env.allowLocalModels = false;

document.addEventListener("DOMContentLoaded", async () => {
    const imageContainer = document.querySelector(".custom-file-upload");
    const status = document.getElementById("status");
    const fileUpload = document.getElementById("upload");
    const resetButton = document.getElementById("reset-image");
    const uploadButton = document.getElementById("upload-btn");
    const container = document.querySelector(".custom-file-upload");

    status.textContent = "Loading model...";

    //const classifier = await pipeline("image-classification", "SemilleroCV/resnet50-finetuned-bwmp2-224");
    const classifier = await pipeline("image-classification", "Factral/test25");

    status.textContent = "Ready";

    async function detect(img) {
        status.textContent = "Analysing...";
        const output = await classifier(img.src, { topk: 0 });
        console.log("output", output);
        status.textContent = "";
        console.log("output", output);
        status.innerHTML = `<strong>Prediction: ${output[0].label}</strong> <br> prob: ${output[0].score}`;
        // output[0].label;
      }

    fileUpload.addEventListener("change", function (e) {
        const file = e.target.files[0];
        if (!file) {
            return;
        };
        const reader = new FileReader();

        reader.onload = function (e2) {
            uploadButton.style.display = 'none';
            const image = document.createElement("img");
            image.src = e2.target.result;
            image.style.maxWidth = "100%"; 
            image.style.height = "100%";
            image.id = "inputImage"
            imageContainer.appendChild(image);
            detect(image);
        };
        reader.readAsDataURL(file);
    });

    resetButton.addEventListener("click", () => {
        const img = document.getElementById("inputImage")
        if(img){
            img.remove();
        }  
        fileUpload.value="";
        status.textContent = "Ready";  
        uploadButton.style.display='flex'; 
    });

    container.addEventListener('click', () => {
        fileUpload.click();
    });
});

