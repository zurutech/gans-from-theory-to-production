let canvas, canvas_context;

function main() {

    console.log("Loading");
    canvas = document.getElementById('canvas');
    canvas_context = canvas.getContext('2d');

    async function generate() {
        console.log("Generating");
        let response = await Model.generate();
        let image = new Image();
        let image_base64 = response.base64_image;
        image.src = 'data:image/png;base64,' + image_base64;
        image.onload = function () {
            canvas_context.drawImage(image, 0, 0, canvas.width, canvas.height);
        };
    }

    document.getElementById("generate").addEventListener("click", generate,
        false);

}