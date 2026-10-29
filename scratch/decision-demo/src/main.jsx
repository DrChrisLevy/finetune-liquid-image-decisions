import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Check, ChevronLeft, ChevronRight, Code2, Download, Focus, Image as ImageIcon, LoaderCircle, Play, RotateCcw, X} from 'lucide-react';
import './style.css';
import {newEvaluation, recordResult, evaluationStats} from './evaluation.js';

const providers = ['liquid', 'liquid3b', 'openai', 'jev'];
const short = {liquid: 'Liquid 2B', liquid3b: 'Liquid 3B', openai: 'OpenAI', jev: 'JEV'};
const ms = n => n == null ? '—' : n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`;
const median = ns => {if (!ns.length) return null; const a = [...ns].sort((x,y)=>x-y); return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
async function api(path, options) {
  const r = await fetch(path, options);
  let b;
  try {b = await r.json();} catch {throw new Error('The local backend is unavailable. Start uv run server.py, then retry.');}
  if (!r.ok) throw new Error(typeof b.detail === 'string' ? b.detail : 'Request failed.');
  return b;
}
function download(value, name) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
  const a=document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}
function candidate(v) {try {return JSON.parse(v);} catch {return {description:v};}}

function App() {
  const [rows,setRows]=useState([]), [config,setConfig]=useState(null), [index,setIndex]=useState(0);
  const [results,setResults]=useState({}), [history,setHistory]=useState([]), [busy,setBusy]=useState(false);
  const [notice,setNotice]=useState('');
  const [tab,setTab]=useState('candidates'), [provider,setProvider]=useState('liquid'), [payload,setPayload]=useState(null);
  const [highlight,setHighlight]=useState(null), [truth,setTruth]=useState(false), [fit,setFit]=useState(true);
  const [clock,setClock]=useState(0), [started,setStarted]=useState(null);
  const imageBox=useRef(null), imageScroll=useRef(null), stopBatch=useRef(false);
  const [batch,setBatch]=useState(false), [batchProgress,setBatchProgress]=useState(0);
  const [evaluation,setEvaluation]=useState(null);
  const row=rows[index], running=busy||batch;
  useEffect(()=> {Promise.all([api('/api/config'),api('/api/examples')]).then(([c,r])=>{setConfig(c);setRows(r);}).catch(e=>setNotice(e.message));},[]);
  useEffect(()=> {if (!row) return; let active=true; setPayload(null); api(`/api/payload/${row.index}/${provider}`).then(p=>{if(active)setPayload(p);}).catch(e=>setNotice(e.message)); return()=>{active=false;};},[index,provider,rows]);
  useEffect(()=> {if(!started)return; const t=setInterval(()=>setClock(performance.now()-started),50);return()=>clearInterval(t);},[started]);
  function choose(i) {setIndex(i);setResults({});setHighlight(null);setTruth(false);setClock(0); if(imageScroll.current)imageScroll.current.scrollTop=0;}
  async function refresh() {const c=await api('/api/config');setConfig(c);return c;}
  async function run(i=index, evaluationId=null, selectedProviders=null) {
    setBusy(true);setNotice('');setHighlight(null);setTruth(false);setResults({});
    try {
      const c=selectedProviders?null:await refresh();
      const active=selectedProviders||providers.filter(p=>c.providers[p].enabled);
      const start=performance.now();setStarted(start);setClock(0);
      setResults(Object.fromEntries(active.map(p=>[p,{pending:true}])));
      await Promise.all(active.map(async p=>{
        try {
          const result=await api(`/api/predict/${p}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:rows[i].index})});
          setResults(old=>({...old,[p]:result}));
          setHistory(old=>[...old,{...result,at:new Date().toISOString()}]);
          if(evaluationId)setEvaluation(old=>recordResult(old,evaluationId,result));
        } catch(e) {
          const result={provider:p,index:rows[i].index,error:e.message};
          setResults(old=>({...old,[p]:result}));
          if(evaluationId)setEvaluation(old=>recordResult(old,evaluationId,result));
        }
      }));
    } catch(e) {setNotice(e.message);} finally {setStarted(null);setBusy(false);}
  }
  async function runBatch() {
    setBatch(true);setBatchProgress(0);stopBatch.current=false;
    const id=crypto.randomUUID();
    try {
      const c=await refresh(), active=providers.filter(p=>c.providers[p].enabled);
      setEvaluation(newEvaluation(id,active,rows.length));
      for(let i=0;i<rows.length;i++){
        if(stopBatch.current)break;
        choose(i);await run(i,id,active);setBatchProgress(i+1);
        setEvaluation(old=>old?.id===id?{...old,completed:i+1}:old);
      }
      setEvaluation(old=>old?.id===id?{...old,status:old.completed===old.total?'complete':'stopped'}:old);
    } catch(e) {
      setNotice(e.message);
      setEvaluation(old=>old?.id===id?{...old,status:'failed'}:old);
    }
    finally{setBatch(false);}
  }
  function reset() {setHistory([]);setResults({});setEvaluation(null);setTruth(false);setHighlight(null);setNotice('');}
  function locate(key) {
    setHighlight(key);setFit(false);
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      const b=candidate(row.question.criteria[key]).bbox;
      if(b&&imageScroll.current&&imageBox.current) {
        const y=b[1]/row.state.screenshot_size_pixels[1]*imageBox.current.clientHeight;
        imageScroll.current.scrollTo({top:Math.max(0,y-imageScroll.current.clientHeight/2),behavior:'smooth'});
      }
    }));
  }
  if(!config||!row)return <main className="loading">{notice?<>{notice}<button onClick={()=>window.location.reload()}>Retry</button></>:<><LoaderCircle className="spin"/> Loading examples…</>}</main>;
  const entries=Object.entries(row.question.criteria), shown=highlight||(truth?row.label:null);
  const bbox=shown?candidate(row.question.criteria[shown]).bbox:null;
  const [w,h]=row.state.screenshot_size_pixels;
  const task=row.state.task_and_past_actions.split('\nPast actions:');
  return <div className="app">
    <h1 className="page-title">Decision Model Testing</h1>
    {notice&&<div className="notice" role="status">{notice}<button className="icon" onClick={()=>setNotice('')} aria-label="Dismiss"><X size={15}/></button></div>}
    <div className="toolbar"><div className="picker"><span className="label">EXAMPLE</span><select value={index} disabled={running} onChange={e=>choose(Number(e.target.value))} aria-label="Choose example">{rows.map((r,i)=><option key={r.id} value={i}>{String(i+1).padStart(2,'0')} · {r.state.task_and_past_actions.split('\n')[0].slice(0,82)}</option>)}</select><button className="icon" aria-label="Previous example" disabled={running||index===0} onClick={()=>choose(index-1)}><ChevronLeft size={18}/></button><button className="icon" aria-label="Next example" disabled={running||index===rows.length-1} onClick={()=>choose(index+1)}><ChevronRight size={18}/></button></div>
      <div className="run-controls">{batch?<button onClick={()=>{stopBatch.current=true;}}>Stop after this example ({batchProgress}/{rows.length})</button>:<button className="quiet batch" disabled={running} onClick={runBatch}>Run {rows.length}</button>}<button className="primary" disabled={running} onClick={()=>run()}>{busy?<LoaderCircle className="spin" size={16}/>:<Play size={16} fill="currentColor"/>} {busy?'Comparing…':'Compare models'}</button></div>
    </div>
    {evaluation&&<section className="evaluation-summary" aria-label="Run accuracy">
      <div className="evaluation-progress"><span className="label">RUN ACCURACY</span><strong>{evaluation.completed}<span> / {evaluation.total} test examples</span></strong><progress max={evaluation.total} value={evaluation.completed}/><small>{({running:'Running',complete:'Complete',stopped:'Stopped',failed:'Failed'})[evaluation.status]} · this run only</small></div>
      {providers.map(p=>{
        const stats=evaluationStats(evaluation,p), enabled=evaluation.providers.includes(p);
        const errors=Object.values(evaluation.results).filter(r=>r.provider===p&&r.error);
        return <div className={`evaluation-model ${p}`} key={p}><span>{config.providers[p].name}</span><strong>{enabled&&stats.accuracy!==null?`${stats.accuracy.toFixed(1)}%`:'—'}</strong><div>{enabled?`${stats.correct} / ${stats.answered} correct`:'Not configured'}</div><small>{enabled?`${ms(stats.median)} median · errors excluded from accuracy`:''}</small>{errors.length>0&&<details className="eval-errors"><summary>{errors.length} failed request{errors.length===1?'':'s'}</summary>{errors.map(r=><p key={r.index}>Example {r.index+1}: {r.error}</p>)}</details>}</div>;
      })}
    </section>}
    <main className="workspace">
      <section className="panel screenshot-panel"><div className="panel-title"><span><ImageIcon size={16}/> Screenshot <span className="muted">{row.image_sizes[0].join(' × ')}</span></span><div><button className={`icon ${truth?'active':''}`} aria-label="Reveal correct element" title="Reveal correct element" onClick={()=>setTruth(!truth)}><Check size={17}/></button></div></div>
        <div className={`image-stage ${fit?'fit':''}`} ref={imageScroll}><div className="image-wrap" ref={imageBox}><img src={row.images[0]} alt={`Mind2Web screenshot for example ${index+1}`} onClick={()=>setFit(!fit)}/>{bbox&&<div className={`bbox ${shown===row.label&&truth?'truth':''}`} style={{left:`${100*bbox[0]/w}%`,top:`${100*bbox[1]/h}%`,width:`${100*bbox[2]/w}%`,height:`${100*bbox[3]/h}%`}}><span>{shown}</span></div>}</div></div>
      </section>
      <section className="right-side"><div className="task panel"><span className="label">THE TASK</span><p>{task[0]}</p>{task[1]?.trim()&&<details><summary>Past actions</summary><pre>{task[1].trim()}</pre></details>}</div>
        <section className="panel input-panel"><div className="tabbar"><div><button className={tab==='candidates'?'selected':''} onClick={()=>setTab('candidates')}>Candidates <span>{entries.length}</span></button><button className={tab==='payload'?'selected':''} onClick={()=>setTab('payload')}><Code2 size={14}/> Request payload</button></div><span className="hint">Label stays out of the input</span></div>
          {tab==='candidates'?<div className="candidate-list">{entries.map(([key,v])=>{const c=candidate(v);return <button key={key} className={`candidate ${shown===key?'selected':''}`} onMouseEnter={()=>setHighlight(key)} onMouseLeave={()=>setHighlight(null)} onFocus={()=>setHighlight(key)} onClick={()=>locate(key)}><code>{key}</code><span className="tag">{c.tag||'element'}</span><span className="candidate-description">{c['aria-label']||c.text||c.description||[c.id,c.type,c.name].filter(Boolean).join(' · ')||`bbox: ${c.bbox?.map(n=>Math.round(n)).join(', ')}`}</span><Focus size={13}/></button>;})}</div>:<div className="payload-area"><div className="payload-tools"><div className="segments">{providers.map(p=><button key={p} className={provider===p?'active':''} onClick={()=>setProvider(p)}>{short[p]}</button>)}</div><button className="quiet" onClick={()=>api(`/api/payload/${row.index}/${provider}?full=true`).then(p=>download(p,`${provider}-payload.json`)).catch(e=>setNotice(e.message))}><Download size={13}/> Full JSON</button></div><pre className="json">{payload?JSON.stringify(payload,null,2):'Loading payload…'}</pre><p className="payload-note">Exact request body; base64 shortened for display. Download includes all image bytes.</p></div>}
        </section>
        <div className="results-heading"><span className="label">LIVE RESULTS</span></div>
        <div className="results">{providers.map(p=>{
          const cfg=config.providers[p],r=results[p], past=history.filter(x=>x.provider===p),unique=Object.values(Object.fromEntries(past.map(x=>[x.index,x]))),n=unique.length,correct=unique.filter(x=>x.correct).length;
          const probs=Object.entries(r?.probabilities||{}).sort((a,b)=>b[1]-a[1]).slice(0,3);
          return <article className={`result ${p}`} key={p}>
            <div className="model-name"><span className="model-mark">{p.startsWith('liquid')?'L':p==='openai'?'O':'J'}</span><strong>{cfg.name}</strong><span className={`status-dot ${cfg.enabled?'on':''}`}/></div><div className="model-detail">{cfg.model}</div>
            <div className={`answer ${r?.error?'error':''}`}>{!cfg.enabled?<><span className="key-needed">API key needed</span><span className="key-needed">Set {cfg.key_name} in .env</span></>:r?.pending?<><span className="pending"><LoaderCircle className="spin" size={17}/> Deciding</span><span className="live-clock">{ms(clock)}</span></>:r?.error?<span>{r.error}</span>:r?.choice?<><div className="decision"><button onClick={()=>locate(r.choice)}>{r.choice}</button><span className={`outcome ${r.correct?'correct':'incorrect'}`}>{r.correct?<Check size={12}/>:<X size={12}/>} {r.correct?'Correct':'Incorrect'}</span></div><div className="latency">{ms(r.latency_ms)}<span>round trip</span></div></>:<><span className="idle">Ready when you are</span><span className="placeholder">—</span></>}</div>
            {probs.length>0&&<div className="probabilities">{probs.map(([key,value])=><button key={key} onClick={()=>locate(key)}><span>{key}</span><div><i style={{width:`${value*100}%`}}/></div><b>{Math.round(value*100)}%</b></button>)}</div>}
            {!evaluation&&<div className="card-stats"><span><b>{n?`${correct}/${n}`:'—'}</b> correct this session</span><span><b>{ms(median(past.map(x=>x.latency_ms)))}</b> median</span></div>}
            {r?.raw&&<details className="response"><summary>Response JSON</summary><pre>{JSON.stringify(r.raw,null,2)}</pre></details>}
          </article>;
        })}</div>
      </section>
    </main>
    <div className="session-actions"><button disabled={running} onClick={reset}><RotateCcw size={14}/> Reset session</button></div>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
