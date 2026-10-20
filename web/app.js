const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");

let drawing = false;
let last = 0; //last position of pointer

function clearCanvas(){
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.clientWidth,canvas.height);
}

clearCanvas();
ctx.strokeStyle = "white";
ctx.lineWidth = 18;      
ctx.lineCap = "round";
ctx.lineJoin = "round";

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    return{
        x : (e.clientX - rect.left) * (canvas.width/rect.width),
        y : (e.clientY - rect.top) * (canvas.height/rect.height),
    };
}

canvas.addEventListener('pointerdown', (e) =>{
    drawing = true;
    canvas.setPointerCapture(e.pointerId);
    last = getPos(e);
})

canvas.addEventListener('pointermove', (e)=>{
    if(!drawing) return;
    
    const p = getPos(e);
    ctx.beginPath();
    ctx.moveTo(last.x,last.y);
    ctx.lineTo(p.x,p.y);
    ctx.stroke();
    last = p;
})

canvas.addEventListener('pointerup', ()=>{ drawing = false;});
canvas.addEventListener('pointercancel', ()=>{ drawing = false;});

//buttons
document.getElementById("clearBtn").addEventListener('click',clearCanvas);
document.getElementById("predictBtn").addEventListener('click',()=> {
    console.log("Predict button clicked");
});
