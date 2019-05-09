

let sequence = [];
let correct = 0;
let total = 0;

// Creat the session and load the pre-trained model
const session = new onnx.InferenceSession();
session.loadModel("./model.onnx");
const input_len = 10;

async function makePrediction(){
    let inputTensor = new onnx.Tensor(sequence.slice(sequence.length-input_len-1, sequence.length-1), 'float32',[1,10]);
    // Run model with Tensor inputs and get the result.
    const outputMap = await session.run([inputTensor]);

    let output = outputMap.values().next().value.data;

    output = output.indexOf(Math.max(...output));
    //console.log("out", output);
    return output;
}

function update(pred){
    var prediction_disp = document.getElementById("input");
    prediction_disp.textContent=`Your input: ${sequence.slice(sequence.length-input_len-1, sequence.length-1)}`;
    var prediction_disp = document.getElementById("last");
    prediction_disp.textContent=`You typed ${sequence[sequence.length-1]}`;

    total += 1;
    if(pred==sequence[sequence.length-1]){
        correct +=1;
    }

    var prediction_disp = document.getElementById("prediction");
    prediction_disp.textContent=`Model predicted ${pred} and it was correct ${100*correct/total}% of the time`;

}

async function zero(){
    //console.log("zero");
    sequence.push(0);
    if(sequence.length>input_len){
        const out = await makePrediction();
        update(out);
    }
}
async function one(){
    //console.log("one");
    sequence.push(1);
    if(sequence.length>input_len){
        const out = await makePrediction();
        update(out);
    }
}

