'use strict';
const ui = Object.fromEntries(['connect','voice','vad','status','record','end','interrupt','disconnect','level','record-time','error','recover','history','metrics','download'].map(id=>[id,document.getElementById(id)]));
const csrf = document.querySelector('meta[name="csrf-token"]').content;
let socket, context, capture, stream, inputNode, silent, connected=false, ready=false, workletLoaded=false;
let turn=null, beginRequest=null, recorderState='idle', flushResolve=null, cancelled=false;
let inputSequence=0, recordedSamples=0, recordingStart=0, voicedMs=0, lastVoice=0;
let sources=new Set(), nextPlay=0, firstPlay=null, endPlay=null, packetCount=0, mediaSamples=0;
let completeEvent=null, ackSent=false, stopAt=null, firstAudioSeconds=null, starvation=0;
let currentArticle=null, bgCount=0, endMode='manual', lastSampleRate=null;
const measurements=[];
const status = text => {ui.status.textContent=text;};
const error = text => {ui.error.textContent=text;};
function buttons() {
  ui.connect.disabled=connected;
  ui.record.disabled=!ready || !connected || recorderState!=='idle';
  ui.end.disabled=recorderState!=='recording';
  ui.interrupt.disabled=!turn && recorderState==='idle';
  ui.disconnect.disabled=!connected; ui.voice.disabled=recorderState!=='idle';
}
async function post(path,body={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify(body)});
  const data=await response.json();
  if (!response.ok) throw new Error(data.message || '请求失败，请刷新并重新登录');
  return data;
}
function send(event) {
  if (!socket || socket.readyState!==WebSocket.OPEN) throw new Error('连接已断开，请重新连接');
  if (socket.bufferedAmount > 65536) throw new Error('网络发送积压，请重新录音');
  socket.send(JSON.stringify(event));
}
function row(question='',reply='') {
  const article=document.createElement('article'), q=document.createElement('p'), a=document.createElement('p');
  q.textContent='你：'+question; a.textContent='助手：'+reply; article.append(q,a); ui.history.append(article);
  // Keep browser history bounded independently of server context.
  while(ui.history.children.length>20) ui.history.firstChild.remove();
  return {article,q,a,text:''};
}
function outputClock() {
  const ts=context.getOutputTimestamp?.();
  if(ts && ts.contextTime>0 && ts.performanceTime>0) return {time:ts.contextTime,wall:ts.performanceTime};
  return {time:context.currentTime-(context.outputLatency||context.baseLatency||0),wall:performance.now()};
}
function stopSources() {
  for(const source of sources) {source.onended=null; try{source.stop();}catch(_){} source.disconnect();}
  sources.clear(); nextPlay=0; firstPlay=null; endPlay=null; completeEvent=null;
}
function stopMic() {
  if(capture) {capture.port.onmessage=null; capture.disconnect(); capture=null;}
  if(inputNode){inputNode.disconnect(); inputNode=null;}
  if(silent){silent.disconnect();silent=null;}
  if(stream){stream.getTracks().forEach(track=>track.stop());stream=null;}
  if(flushResolve){flushResolve();flushResolve=null;}
  recorderState='idle'; ui.level.value=0; buttons();
}
function resetPlayback() {
  stopSources(); packetCount=0;mediaSamples=0;stopAt=null;firstAudioSeconds=null;starvation=0;
  completeEvent=null;ackSent=false;bgCount=0;cancelled=false;
}
function saveMeasurement(record) {
  measurements.push(record); ui.metrics.textContent=JSON.stringify(measurements.slice(-3),null,2);
  post('/dialogue/metrics',record).catch(e=>error('测量保存失败：'+e.message));
}
function interrupt(reason='已打断，可以开始下一轮') {
  const start=performance.now(), old=turn;
  cancelled=true; beginRequest=null; stopMic(); stopSources(); turn=null;
  if(old && socket?.readyState===WebSocket.OPEN) {
    try{send({type:'cancel',turn_id:old.turn_id});}catch(e){error(e.message);}
    saveMeasurement({kind:'interrupt',turn_id:old.turn_id,voice_id:old.voice_id,interrupt_stop_ms:performance.now()-start,complete:false,background_count:bgCount});
  }
  status(reason); buttons();
}
async function connect() {
  error('');
  if(!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('请通过 SSH 转发后打开脚本显示的 http://127.0.0.1 地址，以启用麦克风');
  }
  context=context || new AudioContext(); await context.resume();
  if(!workletLoaded) {
    // Fetch in the authenticated document first. A broken tunnel or expired
    // login should have an actionable error, not Worklet's generic NetworkError.
    let response;
    try {
      response=await fetch('/static/dialogue-capture.js',{
        credentials:'same-origin',cache:'no-store',signal:AbortSignal.timeout(10000)});
    } catch(_) {throw new Error('连接已中断，无法加载录音组件。请检查 SSH 窗口是否出现 Timeout，再重新连接并刷新页面。');}
    if(response.status===401) throw new Error('登录已失效，请刷新页面并重新输入访问口令');
    if(!response.ok || !/javascript/.test(response.headers.get('Content-Type')||''))
      throw new Error('录音组件加载失败，请刷新页面；仍失败时告知返回状态 '+response.status);
    const moduleURL=URL.createObjectURL(new Blob([await response.text()],{type:'application/javascript'}));
    try {await context.audioWorklet.addModule(moduleURL,{credentials:'same-origin'});workletLoaded=true;}
    finally {URL.revokeObjectURL(moduleURL);}
  }
  const ticket=await post('/dialogue/connect');
  socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}${ticket.ws_path}`,['liveact-dialogue-v1',ticket.ws_token]);
  socket.onopen=()=>{connected=true;status('正在恢复会话');buttons();};
  socket.onmessage=event=>{try{receive(JSON.parse(event.data));}catch(e){error(e.message);interrupt('播放或协议错误，请重新录音');}};
  socket.onerror=()=>error('连接失败，请检查 SSH 隧道与作业是否仍运行');
  socket.onclose=()=>{connected=false;ready=false;interrupt('连接已断开，未完成的回答已取消；点击连接重试');buttons();};
}
async function startRecording() {
  if(turn) interrupt();
  await context.resume(); resetPlayback();error('');
  recorderState='starting';beginRequest=crypto.randomUUID();status('等待录音准备');buttons();
  send({type:'begin',request_id:beginRequest,voice_id:ui.voice.value});
}
async function openMic(event) {
  turn=event; const expected=event.turn_id;
  try {
    const acquired=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    if(cancelled || !turn || turn.turn_id!==expected || !connected){acquired.getTracks().forEach(t=>t.stop());return;}
    stream=acquired; inputNode=context.createMediaStreamSource(stream);
    capture=new AudioWorkletNode(context,'dialogue-capture'); silent=context.createGain();silent.gain.value=0;
    inputNode.connect(capture);capture.connect(silent).connect(context.destination);
    lastSampleRate=context.sampleRate;inputSequence=0;recordedSamples=0;voicedMs=0;lastVoice=0;recordingStart=performance.now();
    capture.port.onmessage=message=>{
      if(message.data.type==='flushed') {const done=flushResolve;flushResolve=null;done?.();return;}
      if(!turn || turn.turn_id!==expected || !['recording','ending'].includes(recorderState))return;
      const {buffer,rms}=message.data;
      const bytes=new Uint8Array(buffer), samples=bytes.length/2;
      if(recordedSamples+samples>240000){endRecording('limit').catch(fail);return;}
      recordedSamples+=samples;ui.level.value=rms;ui['record-time'].textContent=(recordedSamples/16000).toFixed(1)+' 秒';
      try {
        const encoded=btoa(String.fromCharCode(...bytes));
        send({type:'input',turn_id:turn.turn_id,sequence:inputSequence++,pcm:encoded});
      } catch(e){fail(e);interrupt('网络积压，录音已取消');return;}
      if(rms>.012){voicedMs+=samples/16;lastVoice=performance.now();}
      if(recordedSamples>=240000) endRecording('limit').catch(fail);
      else if(ui.vad.checked && voicedMs>=250 && lastVoice && performance.now()-lastVoice>=1000 && recorderState==='recording') endRecording('vad').catch(fail);
    };
    recorderState='recording';currentArticle=row('正在录音…','');status('聆听中（最多 15 秒）');buttons();
  } catch(e){interrupt('麦克风未启用');error('麦克风错误：'+e.message+'；请允许此页面使用麦克风');}
}
async function endRecording(mode='manual') {
  if(recorderState!=='recording')return;
  recorderState='ending';endMode=mode;stopAt=performance.now();buttons();status('正在结束录音');
  const expected=turn?.turn_id;
  await new Promise(resolve=>{flushResolve=resolve;capture.port.postMessage('flush');});
  if(!turn || turn.turn_id!==expected || cancelled)return;
  stopMic();
  if(recordedSamples<8000){interrupt('录音不足 0.5 秒，请重新开始');return;}
  send({type:'end',turn_id:turn.turn_id});status('正在识别和思考');buttons();
}
function audio(event) {
  if(event.sequence!==packetCount || event.sample_start!==mediaSamples || event.sample_rate!==16000) throw new Error('回答音频样本顺序不一致');
  const raw=atob(event.pcm); if(!raw.length || raw.length%2 || raw.length>3200)throw new Error('回答音频格式无效');
  const bytes=Uint8Array.from(raw,c=>c.charCodeAt(0)), view=new DataView(bytes.buffer);
  if(mediaSamples+bytes.length/2>240000) throw new Error('回答超过 15 秒限制');
  const buffer=context.createBuffer(1,bytes.length/2,16000), channel=buffer.getChannelData(0);
  for(let i=0;i<channel.length;i++)channel[i]=view.getInt16(i*2,true)/32768;
  const now=context.currentTime;
  const start=Math.max(nextPlay,now+.03);
  if(firstPlay!==null && start>nextPlay)starvation+=start-nextPlay;
  if(firstPlay===null)firstPlay=start;
  const source=context.createBufferSource();source.buffer=buffer;source.connect(context.destination);sources.add(source);
  source.onended=()=>{sources.delete(source);source.disconnect();};source.start(start);
  nextPlay=start+buffer.duration;endPlay=nextPlay;packetCount++;mediaSamples+=channel.length;status('正在回答');
}
function receive(event) {
  if(event.type==='reset' || event.type==='service_error') {
    ready=false;interrupt(event.type==='reset'?'语音服务正在恢复，上下文已清空':'语音服务启动失败，请点击恢复');ui.history.replaceChildren();return;
  }
  if(event.type==='ready' || event.type==='attached') {
    ready=true;turn=null;recorderState='idle';ui.history.replaceChildren();
    for(const [q,a] of (event.history||[]))row(q,a);
    status('已就绪，可以开始录音');buttons();return;
  }
  if(event.type==='begun') {
    if(event.request_id!==beginRequest){try{send({type:'cancel',turn_id:event.turn_id});}catch(_){}return;}
    beginRequest=null;openMic(event);return;
  }
  if(event.type==='command_error'){error(event.message);return;}
  if(!turn || event.turn_id!==turn.turn_id || event.epoch!==turn.epoch || cancelled)return;
  if(event.type==='recognized')currentArticle.q.textContent='你：'+(event.text || '（未识别到语音）');
  else if(event.type==='text'){currentArticle.text+=event.text;currentArticle.a.textContent='助手：'+currentArticle.text;}
  else if(event.type==='audio')audio(event);
  else if(event.type==='complete') {
    if(event.audio_packets!==packetCount || event.audio_samples!==mediaSamples)throw new Error('回答结束标记不匹配');
    completeEvent=event;
  } else if(event.type==='committed') {turn=null;status('回答播放完毕，可以开始下一轮');buttons();}
  else if(event.type==='no_speech'){turn=null;status('未检测到有效语音，请重新录音');buttons();}
  else if(event.type==='error' || event.type==='cancelled'){error(event.reason||'本轮已取消');interrupt('本轮已结束，可以重试');}
  else if(event.type==='stage' && firstPlay===null)status({recognizing:'正在识别',answering:'正在思考',synthesizing:'正在合成语音'}[event.stage]||'正在处理');
}
setInterval(()=>{
  if(!context || !turn || cancelled || firstPlay===null)return;
  const clock=outputClock();
  if(firstAudioSeconds===null && clock.time>=firstPlay && stopAt!==null) {
    const wall=clock.wall-(clock.time-firstPlay)*1000;
    firstAudioSeconds=Math.max(0,(wall-stopAt)/1000);
  }
  if(completeEvent && endPlay!==null && clock.time>=endPlay && !ackSent) {
    ackSent=true;
    saveMeasurement({kind:'turn',turn_id:turn.turn_id,voice_id:turn.voice_id,end_mode:endMode,
      stop_to_first_audio_s:firstAudioSeconds,media_seconds:mediaSamples/16000,
      play_wall_seconds:Math.max(0,clock.time-firstPlay),starvation_seconds:starvation,
      packet_count:packetCount,sample_rate:lastSampleRate,background_count:bgCount,complete:true});
    send({type:'ack',turn_id:turn.turn_id,last_audio_sequence:packetCount-1});
  }
},25);
function fail(e){error(e.message||String(e));}
ui.connect.onclick=()=>connect().catch(fail);
ui.record.onclick=()=>startRecording().catch(e=>{stopMic();fail(e);});
ui.end.onclick=()=>endRecording().catch(fail);
ui.interrupt.onclick=()=>interrupt();
ui.disconnect.onclick=()=>{interrupt('已断开');socket?.close();};
ui.recover.onclick=async()=>{interrupt('正在恢复语音服务');ready=false;buttons();try{await post('/dialogue/recover');}catch(e){fail(e);}};
ui.download.onclick=()=>{const blob=new Blob([JSON.stringify(measurements,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='liveact-dialogue-metrics.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.addEventListener('visibilitychange',()=>{if(document.hidden && turn){bgCount++;interrupt('页面已切到后台，本轮已取消；返回后重新录音');}});
window.addEventListener('pagehide',()=>{stopMic();stopSources();socket?.close();});
buttons();
