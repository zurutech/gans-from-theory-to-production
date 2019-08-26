var canvas, canvas_context;

function main() {

    console.log("Loading");
    canvas = document.getElementById('canvas');
    context = canvas.getContext('2d');

    async function generate() {
        console.log("Generating");
        response = await Model.generate();
        var image = new Image();
        image_base64 = response.base64_image
        image.src = 'data:image/png;base64,' + image_base64;
        image.onload = function () {

            context.drawImage(image, 0, 0, canvas.width, canvas.height);
        };
    };

    document.getElementById("generate").addEventListener("click", generate,
        false);

}