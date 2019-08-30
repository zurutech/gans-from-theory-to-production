/**
 * @fileoverview Machine-learning related methods. Contains the predict method.
 */

/** @namespace */
var Model = Model || {};

// Remote parsing of generated image
let model_url =
    "https://us-central1-machine-learning-199407.cloudfunctions.net/euroscipy-2019-tutorial-128";

Model.generate = async function (noise_vector) {
    let data;

    console.log("Generating");

    if (noise_vector == null) {
        data = {};
    } else {
        data = {
            "noise_vector": noise_vector
        };
    }
    console.log("Sending data:", data);

    // send the request to endpoint
    let response = await $.ajax({
        url: model_url,
        type: "POST",
        data: JSON.stringify(data),
        processData: false,
        contentType: "application/json",
        dataType: "json"
    });
    console.log("Received response from gcloud", response);

    // return the response
    return {
        "base64_image": response.base64_image,
        "noise_vector": response.noise_vector
    };
};
