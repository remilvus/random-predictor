const session = new onnx.InferenceSession();
session.loadModel("./model.onnx");
const input_len = 10;
const showed_input_len = 10;

let evaluated = input_len - 1; //index of last evaluated value
let table_end = 0;
let predictions = Array(input_len).fill('-');
let user_input = [];
let correct = 0;
let total = 0;
let isEvaluating = false; //is true when evaluating after every user input

function zero(){
    user_input.push(0);
    if(isEvaluating)
        make_predictions();
    show_input()
}

function one(){
    user_input.push(1);
    if(isEvaluating)
        make_predictions();
    show_input()
}

function show_input(){
    let input_display = document.getElementById("input");
    let start = user_input.length-showed_input_len - 1;
    if (start < 0) start = 0;
    input_display.innerHTML = `Your sequence length is ${user_input.length}. <br>
                                 End of your input: ${user_input.slice(start, user_input.length-1)}`;
}

function show_pred(){
    let correctness_display = document.getElementById("eval");
    correctness_display.textContent = `The model predicted correnctly ${correct} out of ${total}. (${(100*correct/total).toFixed(2)}%)`
}

function make_table(){
    let table = document.getElementById("sequences_table");
    if(table_end==0){
        let head = table.insertRow();
        head.innerHTML = `<th>Your sequence</th>
                         <th>Model predictions</th>`
    }
    while(table_end < user_input.length - 1){
        table_end++;
        let row = table.insertRow();
        let cell = row.insertCell();
        let text = document.createTextNode(user_input[table_end]);
        cell.appendChild(text);
        cell = row.insertCell();
        text = document.createTextNode(predictions[table_end]);
        cell.appendChild(text);
    }
}

async function make_predictions(){
    if(user_input.length > input_len){
        if(isEvaluating){
            evaluated += 1;
            let out = await make_prediction()
            total += 1;
            if (user_input[evaluated] == out) correct += 1;
            show_pred();
            make_table();
        } else {
            while(evaluated < user_input.length - 1){
                evaluated += 1;
                let out = await make_prediction()
                total += 1;
                if (user_input[evaluated] == out) correct += 1;
            }
            show_pred();
            make_table();
        }

    }
}

async function make_prediction(){
    let inputTensor = new onnx.Tensor(user_input.slice(evaluated-input_len, evaluated), 'float32',[1,10]);
    const outputMap = await session.run([inputTensor]);
    let output = outputMap.values().next().value.data;
    output = output.indexOf(Math.max(...output));
    predictions.push(output);
    return output;
}

function change_eval_type(){
    isEvaluating = !isEvaluating;
    total = 0;
    correct = 0;
    user_input = [];
    predictions = Array(input_len).fill('-');
    evaluated = input_len - 1;
    table_end = 0;
    let table = document.getElementById("sequences_table");
    while(table.hasChildNodes())table.removeChild(table.firstChild);

    let eval_b = document.getElementById("eval_button");
    if(isEvaluating) eval_b.style.display='none'
    else eval_b.style.display='block'
}
