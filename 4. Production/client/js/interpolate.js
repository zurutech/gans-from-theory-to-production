let canvas_a, canvas_context_a, canvas_b, canvas_context_b,
    canvas_interpolation, context_interpolation;
let noise_vector_a, noise_vector_b;
let slider;

function main() {

    console.log("Loading");
    canvas_a = document.getElementById('canvas_a');
    canvas_context_a = canvas_a.getContext('2d');
    canvas_b = document.getElementById('canvas_b');
    canvas_context_b = canvas_b.getContext('2d');
    canvas_interpolation = document.getElementById('canvas_interpolation');
    context_interpolation = canvas_interpolation.getContext('2d');
    slider = document.getElementById('customRange1');

    // disable slider
    slider.disabled = true;

    async function generate(canvas, context, noise_vector) {
        // call model
        let response = await Model.generate(noise_vector);

        // get image
        let image = new Image();
        let image_base64 = response.base64_image;

        // get noise vector
        noise_vector = response.noise_vector;

        // display image
        image.src = 'data:image/png;base64,' + image_base64;
        image.onload = function () {
            context.drawImage(image, 0, 0, canvas.width, canvas
                .height);
        };

        return noise_vector;
    }

    async function generate_a() {
        console.log("Generating A");
        noise_vector_a = await generate(canvas_a, canvas_context_a);

        if (noise_vector_b != null) {
            slider.disabled = false;
        }
    }

    async function generate_b() {
        console.log("Generating B");
        noise_vector_b = await generate(canvas_b, canvas_context_b);

        if (noise_vector_a != null) {
            slider.disabled = false;
        }
    }

    document.getElementById("generate_a").addEventListener("click", generate_a,
        false);
    document.getElementById("generate_b").addEventListener("click", generate_b,
        false);

    async function interpolate() {
        let value = slider.value / 100;
        console.log(value);
        let a = nj.array(noise_vector_a);
        let b = nj.array(noise_vector_b);
        let noise_vector_interpolated = a.multiply(1 - value).add(b.multiply(
            value));
        await generate(canvas_interpolation, context_interpolation,
            noise_vector_interpolated.tolist());
    }

    slider.addEventListener("change", interpolate, false);

}