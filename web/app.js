const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");
const preview = document.getElementById("preview");
const pctx = preview.getContext("2d");

let drawing = false;
let last = 0; //last position of pointer

function clearCanvas(){
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.width,canvas.height);
    pctx.fillStyle = "black";
    pctx.fillRect(0,0,28,28);
}

clearCanvas();
ctx.strokeStyle = "white";
ctx.lineWidth = 18;      
ctx.lineCap = "round";
ctx.lineJoin = "round";


// we use getPos to scale the position of the pixel in the displayed CSS grid to 280*280 grid
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

// ---- converting the drawn image to 784 values ---

// testing pixel data format
function px(x,y) {                                      // here x and y are coordinates not row and col
    const d = ctx.getImageData(0,0,280,280).data;
    const i = (y*280 + x) * 4;
    return [d[i],d[i+1],d[i+2],d[i+3]];
}

pctx.imageSmoothingEnabled = true;
pctx.imageSmoothingQuality = 'high';

const INK_THRESHOLD = 40;     // out of 0-255, if value is >40, then ink considered
const TARGET = 22;    // out of 28 pixels in preview, 22 max are filled

function getInput(){
    const w = canvas.width;
    const h = canvas.height;
    const data = ctx.getImageData(0,0,w,h).data;

    // finding the minimum possible box surrounding the drawn figure
    let minX = w;
    let minY = h;
    let maxX = -1;
    let maxY = -1;

    for(let i=0;i<w;i++)
    {
        for(let j=0;j<h;j++)
        {
            let ink = data[(j*w + i)*4];
            if(ink > INK_THRESHOLD)
            {
                if(minX>i) minX = i;
                if(minY>j) minY = j;
                if(maxX<i) maxX = i;
                if(maxY<j) maxY = j;
            }
        }
    }
    if(maxX < 0) return null;  // no ink was found

    const boundWidth = maxX - minX + 1;
    const boundHeight = maxY - minY + 1;

    // scaling dimensions to 28*28 and target(here 22 pixels)
    const scale = TARGET/Math.max(boundHeight,boundWidth);
    const scaledWidth = Math.max(1,Math.round(boundWidth * scale));
    const scaledHeight = Math.max(1,Math.round(boundHeight * scale));

    // starting top left corner of the drawn image in 28*28 pixel grid
    const startX = Math.round((28-scaledWidth)/2);
    const startY = Math.round((28-scaledHeight)/2);

    // draw the image on the preview(28*28) grid
    pctx.fillStyle = "black";
    pctx.fillRect(0,0,28,28);
    pctx.drawImage(canvas, minX, minY, boundWidth, boundHeight, startX, startY, scaledWidth, scaledHeight);

    // read the pixels of preview as 784 numbers
    const previewData = pctx.getImageData(0,0,28,28).data;
    let pixels = new Array(784);
    for(let i=0;i<784;i++)
    {
        pixels[i] = previewData[i*4]/255;    // we only consider the red value(as we are drawing in grayscale) and scale it to 0-1
    }

    return pixels;
}

document.getElementById("predictBtn").addEventListener('click', () => {
  const pixels = getInput();
  if (pixels === null) {
    console.log("Nothing drawn");
    return;
  }
  console.log(pixels.length, pixels);
});