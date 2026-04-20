'use strict';
// ── Constants ────────────────────────────────────────────────
const PINCH_THRESH=0.07,HOLD_SECS=0.5,TAP_MAX=0.30,DBL_SECS=0.40,COOLDOWN=0.50;
const BANNER_SECS=1.2,SMOOTH=0.45,MIN_PX=4,ERASER_R=40,UNDO_MAX=30,DEBOUNCE=3;

// ── DOM ──────────────────────────────────────────────────────
const video=document.getElementById('webcam');
const dc=document.getElementById('dc');
const oc=document.getElementById('oc');
const dctx=dc.getContext('2d');
const octx=oc.getContext('2d');
const mlabel=document.getElementById('mlabel');
const fpslabel=document.getElementById('fpslabel');
const status=document.getElementById('status');
const banner=document.getElementById('banner');

// ── App state ────────────────────────────────────────────────
let color='#ff0000',brush=6;
let prevPt=null,smoothPt=null;
const undoStack=[],redoStack=[];
let pinchActive=false,pinchStart=0,tapCount=0,lastTapEnd=0,lastAction=0,isMoveActive=false;
let moveAnchor=null,moveBase={x:0,y:0},cvOff={x:0,y:0};
let inZoom=false,zInitD=0,zInitS=1,zScale=1;
let mCand='IDLE',mStreak=0,aMode='IDLE';
let strokePts=[],wasDrawing=false;
let lastFT=performance.now(),fCount=0;
let bannerTmr=null;

// ── Toolbar ──────────────────────────────────────────────────
document.querySelectorAll('.cbtn').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.cbtn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); color=b.dataset.c;
}));
document.querySelectorAll('.bbtn').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.bbtn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); brush=+b.dataset.s;
}));
document.getElementById('bundo').onclick=doUndo;
document.getElementById('bredo').onclick=doRedo;
document.getElementById('bclear').onclick=doClear;
document.getElementById('bsave').onclick=doSave;
document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key==='z')doUndo();
  if(e.ctrlKey&&(e.key==='y'||e.key==='Z'))doRedo();
});

// ── Canvas sizing ────────────────────────────────────────────
function syncSize(){
  const wrap=document.getElementById('wrap');
  const w=wrap.clientWidth,h=wrap.clientHeight;
  if(w<1||h<1)return;
  [dc,oc].forEach(c=>{
    if(c.width!==w||c.height!==h){
      let saved=null;
      if(c===dc&&c.width>0&&c.height>0)saved=dctx.getImageData(0,0,c.width,c.height);
      c.width=w;c.height=h;
      if(saved)dctx.putImageData(saved,0,0);
    }
  });
}
new ResizeObserver(syncSize).observe(document.getElementById('wrap'));
window.addEventListener('resize',syncSize);

// ── Undo/Redo ────────────────────────────────────────────────
function snap(){
  undoStack.push(dctx.getImageData(0,0,dc.width,dc.height));
  if(undoStack.length>UNDO_MAX)undoStack.shift();
  redoStack.length=0;
}
function doUndo(){
  if(!undoStack.length)return;
  redoStack.push(dctx.getImageData(0,0,dc.width,dc.height));
  dctx.putImageData(undoStack.pop(),0,0);
  flash('UNDO','undo');
}
function doRedo(){
  if(!redoStack.length)return;
  undoStack.push(dctx.getImageData(0,0,dc.width,dc.height));
  dctx.putImageData(redoStack.pop(),0,0);
  flash('REDO','redo');
}
function doClear(){snap();dctx.clearRect(0,0,dc.width,dc.height);cvOff={x:0,y:0};zScale=1;applyXform();}
function doSave(){const a=document.createElement('a');a.download='drawing_'+Date.now()+'.png';a.href=dc.toDataURL();a.click();}
function flash(txt,cls){
  banner.textContent=txt;banner.className='show '+cls;
  clearTimeout(bannerTmr);bannerTmr=setTimeout(()=>banner.className='',BANNER_SECS*1000);
}
function applyXform(){
  dc.style.transform=`translate(${cvOff.x}px,${cvOff.y}px) scale(${zScale})`;
  dc.style.transformOrigin='center center';
}

// ── Landmark helper ──────────────────────────────────────────
function lm(lms,i){
  return{x:(1-lms[i].x)*oc.width, y:lms[i].y*oc.height};
}
function dst(a,b){return Math.hypot(a.x-b.x,a.y-b.y);}
function pinching(lms){return dst(lm(lms,4),lm(lms,8))/oc.width<PINCH_THRESH;}
function fup(lms,t,p){return lms[t].y<lms[p].y;}

// ── Gesture classify ─────────────────────────────────────────
function classify(lms){
  const th=fup(lms,4,3),ix=fup(lms,8,6),mi=fup(lms,12,10),ri=fup(lms,16,14),pi=fup(lms,20,18);
  if(th&&ix&&mi&&ri&&pi)return'ERASE';
  if(pinching(lms)&&!mi&&!ri&&!pi)return'MOVE';
  if(ix&&!th&&!mi&&!ri&&!pi)return'DRAW';
  if(ix&&mi&&!ri&&!pi)return'CURSOR';
  return'IDLE';
}
function debounce(raw){
  if(raw===mCand)mStreak++;else{mCand=raw;mStreak=1;}
  if(mStreak>=DEBOUNCE)aMode=mCand;
  return aMode;
}

// ── Pinch tap/hold ───────────────────────────────────────────
function updatePinch(p){
  const now=performance.now()/1000;
  if(p&&!pinchActive){pinchActive=true;pinchStart=now;}
  else if(p&&pinchActive){
    if(now-pinchStart>=HOLD_SECS&&!isMoveActive){isMoveActive=true;tapCount=0;flash('MOVE MODE','move');}
  }else if(!p&&pinchActive){
    const dur=now-pinchStart;pinchActive=false;
    if(isMoveActive){isMoveActive=false;moveAnchor=null;}
    else if(dur<TAP_MAX){
      if(tapCount===1&&now-lastTapEnd<DBL_SECS){
        if(now-lastAction>=COOLDOWN){doRedo();lastAction=now;}tapCount=0;
      }else{tapCount=1;lastTapEnd=now;}
    }
  }
  if(!p&&tapCount===1&&performance.now()/1000-lastTapEnd>=DBL_SECS){
    if(performance.now()/1000-lastAction>=COOLDOWN){doUndo();lastAction=performance.now()/1000;}
    tapCount=0;
  }
}

// ── Shape detection ──────────────────────────────────────────
function detectShape(pts){
  if(pts.length<20)return null;
  const sm=pts.map((p,i)=>{
    let sx=0,sy=0,n=0;
    for(let j=Math.max(0,i-2);j<=Math.min(pts.length-1,i+2);j++){sx+=pts[j].x;sy+=pts[j].y;n++;}
    return{x:sx/n,y:sy/n};
  });
  const xs=sm.map(p=>p.x),ys=sm.map(p=>p.y);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const w=x1-x0,h=y1-y0;
  if(w*h<1000)return null;
  let peri=0;for(let i=1;i<sm.length;i++)peri+=dst(sm[i-1],sm[i]);
  const ap=dp(sm,0.03*peri),n=ap.length;
  let shape;
  if(n<=2)shape='LINE';
  else if(n===3)shape='TRIANGLE';
  else if(n===4)shape=(w/h>0.85&&w/h<1.15)?'SQUARE':'RECTANGLE';
  else if(n>6){
    let area=0;for(let i=0,j=sm.length-1;i<sm.length;j=i++)area+=(sm[j].x+sm[i].x)*(sm[j].y-sm[i].y);
    shape=Math.abs(area/2)/(Math.PI*(Math.max(w,h)/2)**2)>0.6?'CIRCLE':'POLYGON';
  }else shape='POLYGON';
  return{shape,ap,x0,y0,w,h,sm};
}
function dp(pts,eps){
  if(pts.length<=2)return pts;
  let mx=0,mi=0;
  for(let i=1;i<pts.length-1;i++){const d=pld(pts[i],pts[0],pts[pts.length-1]);if(d>mx){mx=d;mi=i;}}
  if(mx>eps){const l=dp(pts.slice(0,mi+1),eps),r=dp(pts.slice(mi),eps);return[...l.slice(0,-1),...r];}
  return[pts[0],pts[pts.length-1]];
}
function pld(p,a,b){
  const dx=b.x-a.x,dy=b.y-a.y;
  if(!dx&&!dy)return dst(p,a);
  const t=((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy);
  return dst(p,{x:a.x+t*dx,y:a.y+t*dy});
}
function drawShape(r){
  const{shape,ap,x0,y0,w,h,sm}=r;
  dctx.strokeStyle=color;dctx.lineWidth=brush;dctx.lineCap='round';dctx.lineJoin='round';
  dctx.clearRect(x0-brush-8,y0-brush-8,w+brush*2+16,h+brush*2+16);
  dctx.beginPath();
  if(shape==='LINE'){dctx.moveTo(sm[0].x,sm[0].y);dctx.lineTo(sm[sm.length-1].x,sm[sm.length-1].y);}
  else if(shape==='CIRCLE'){dctx.arc(x0+w/2,y0+h/2,Math.max(w,h)/2,0,Math.PI*2);}
  else if(shape==='RECTANGLE'||shape==='SQUARE'){dctx.rect(x0,y0,w,h);}
  else{dctx.moveTo(ap[0].x,ap[0].y);for(let i=1;i<ap.length;i++)dctx.lineTo(ap[i].x,ap[i].y);dctx.closePath();}
  dctx.stroke();flash(shape,'shape');
}

// ── Skeleton ─────────────────────────────────────────────────
function skeleton(lms){
  const conn=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  octx.strokeStyle='rgba(255,255,255,0.3)';octx.lineWidth=1.5;
  conn.forEach(([a,b])=>{
    const pa=lm(lms,a),pb=lm(lms,b);
    octx.beginPath();octx.moveTo(pa.x,pa.y);octx.lineTo(pb.x,pb.y);octx.stroke();
  });
  for(let i=0;i<21;i++){
    const p=lm(lms,i);
    octx.beginPath();octx.arc(p.x,p.y,3,0,Math.PI*2);
    octx.fillStyle=i===8?'#64ff96':'rgba(255,255,255,0.6)';octx.fill();
  }
}
function setMode(t,c){mlabel.textContent=t;mlabel.style.color=c;}
function finishStroke(){
  if(wasDrawing&&strokePts.length>=20){const r=detectShape(strokePts);if(r)drawShape(r);}
  wasDrawing=false;strokePts=[];
}

// ── MediaPipe result handler ──────────────────────────────────
function onResults(res){
  fCount++;
  const now=performance.now();
  if(now-lastFT>=1000){fpslabel.textContent=Math.round(fCount*1000/(now-lastFT))+' FPS';fCount=0;lastFT=now;}

  const W=oc.width,H=oc.height;
  octx.clearRect(0,0,W,H);
  const hs=res.multiHandLandmarks||[];

  // TWO HANDS → ZOOM
  if(hs.length===2){
    const p0=pinching(hs[0]),p1=pinching(hs[1]);
    if(p0&&p1){
      const t0=lm(hs[0],4),i0=lm(hs[0],8),t1=lm(hs[1],4),i1=lm(hs[1],8);
      const m0={x:(t0.x+i0.x)/2,y:(t0.y+i0.y)/2},m1={x:(t1.x+i1.x)/2,y:(t1.y+i1.y)/2};
      const cd=dst(m0,m1);
      if(!inZoom){inZoom=true;zInitD=cd;zInitS=zScale;}
      zScale=Math.max(0.5,Math.min(3,zScale*.8+(zInitS*cd/zInitD)*.2));
      applyXform();
      octx.strokeStyle='#00ddff';octx.lineWidth=2;
      octx.beginPath();octx.moveTo(m0.x,m0.y);octx.lineTo(m1.x,m1.y);octx.stroke();
      [m0,m1].forEach(p=>{octx.beginPath();octx.arc(p.x,p.y,10,0,Math.PI*2);octx.fillStyle='#00ddff';octx.fill();});
      octx.fillStyle='#00ddff';octx.font='bold 18px sans-serif';
      octx.fillText('Zoom '+Math.round(zScale*100)+'%',(m0.x+m1.x)/2-40,(m0.y+m1.y)/2-16);
      setMode('ZOOM','#00ddff');
    }else inZoom=false;
    prevPt=null;smoothPt=null;return;
  }
  inZoom=false;

  // ONE HAND
  if(hs.length===1){
    const lms=hs[0];
    const mode=debounce(classify(lms));
    const it=lm(lms,8),tt=lm(lms,4);
    if(!smoothPt)smoothPt={...it};
    else{smoothPt.x+=(it.x-smoothPt.x)*(1-SMOOTH);smoothPt.y+=(it.y-smoothPt.y)*(1-SMOOTH);}
    const ix=smoothPt.x,iy=smoothPt.y;
    skeleton(lms);

    if(mode==='ERASE'){
      const pp=[0,4,8,12,16,20].map(i=>lm(lms,i));
      const cx=pp.reduce((s,p)=>s+p.x,0)/pp.length,cy=pp.reduce((s,p)=>s+p.y,0)/pp.length;
      snap();dctx.clearRect(cx-ERASER_R,cy-ERASER_R,ERASER_R*2,ERASER_R*2);
      octx.beginPath();octx.arc(cx,cy,ERASER_R,0,Math.PI*2);octx.strokeStyle='#ff4444';octx.lineWidth=2;octx.stroke();
      setMode('ERASE','#ff5555');prevPt=null;finishStroke();

    }else if(mode==='MOVE'){
      updatePinch(pinching(lms));
      const mx=(ix+tt.x)/2,my=(iy+tt.y)/2;
      if(isMoveActive){
        if(!moveAnchor){moveAnchor={x:mx,y:my};moveBase={...cvOff};}
        else{cvOff.x=moveBase.x+(mx-moveAnchor.x);cvOff.y=moveBase.y+(my-moveAnchor.y);applyXform();}
      }
      octx.beginPath();octx.arc(mx,my,14,0,Math.PI*2);octx.fillStyle='rgba(255,160,50,.8)';octx.fill();
      octx.beginPath();octx.moveTo(tt.x,tt.y);octx.lineTo(ix,iy);octx.strokeStyle='#ffaa44';octx.lineWidth=2;octx.stroke();
      setMode('MOVE','#ffaa44');prevPt=null;finishStroke();

    }else if(mode==='DRAW'){
      updatePinch(false);setMode('DRAW','#64ff96');
      octx.beginPath();octx.arc(ix,iy,brush/2+3,0,Math.PI*2);octx.fillStyle=color;octx.fill();
      octx.strokeStyle='#fff';octx.lineWidth=1;octx.stroke();
      if(!prevPt){prevPt={x:ix,y:iy};strokePts=[{x:ix,y:iy}];snap();}
      else{
        const d=dst({x:ix,y:iy},prevPt);
        if(d>=MIN_PX){
          dctx.beginPath();dctx.moveTo(prevPt.x,prevPt.y);dctx.lineTo(ix,iy);
          dctx.strokeStyle=color;dctx.lineWidth=brush;dctx.lineCap='round';dctx.lineJoin='round';dctx.stroke();
          prevPt={x:ix,y:iy};strokePts.push({x:ix,y:iy});
        }
      }
      wasDrawing=true;

    }else if(mode==='CURSOR'){
      updatePinch(false);setMode('CURSOR','#ffdd44');
      octx.beginPath();octx.arc(ix,iy,12,0,Math.PI*2);octx.strokeStyle='#ffdd44';octx.lineWidth=2;octx.stroke();
      octx.beginPath();octx.arc(ix,iy,3,0,Math.PI*2);octx.fillStyle='#ffdd44';octx.fill();
      prevPt=null;finishStroke();

    }else{
      updatePinch(false);setMode('IDLE','#888');prevPt=null;finishStroke();
    }
  }else{
    prevPt=null;smoothPt=null;updatePinch(false);finishStroke();setMode('IDLE','#888');
  }
}

// ── MediaPipe Hands ───────────────────────────────────────────
const hands=new Hands({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`});
hands.setOptions({maxNumHands:2,modelComplexity:1,minDetectionConfidence:0.75,minTrackingConfidence:0.65});
hands.onResults(onResults);

// ── Start camera ──────────────────────────────────────────────
async function start(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'},audio:false});
    video.srcObject=stream;
    video.onloadedmetadata=()=>{video.play();syncSize();};
    video.onplaying=syncSize;

    // Use MediaPipe Camera utility to feed frames
    const cam=new Camera(video,{
      onFrame:async()=>{
        syncSize();
        await hands.send({image:video});
      },
      width:1280,height:720
    });
    cam.start();

    const track=stream.getVideoTracks()[0];
    status.textContent='Camera: '+(track?track.label:'ready')+' — raise index finger to draw!';
    setTimeout(()=>status.classList.add('hidden'),4000);
  }catch(e){
    status.textContent='Camera error: '+e.message;
    status.style.color='#f55';
    console.error(e);
  }
}
start();
