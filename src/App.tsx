import { useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { Channel, invoke } from '@tauri-apps/api/core';
import {
  Check, CircleAlert, Copy, Database, FolderOpen, Globe2, History, Layers3, Play,
  RefreshCw, Search, Settings2, Sparkles, Terminal, WandSparkles, X
} from 'lucide-react';
import type {
  Constraints, DemographicLevel, DemographicPrompts, GenerationRecord, LibrarySnapshot,
  LlmSettings, PreparedGeneration, PromptPair, ProviderKind, WebHostInfo
} from './types';

type Stage = 'library' | 'compatibility' | 'selection' | 'llm' | 'workflow' | 'comfy' | 'recorded';
type Status = 'idle' | 'running' | 'done' | 'error';

const isTauriRuntime =
  typeof window !== 'undefined' &&
  (!!(window as Window & {__TAURI_INTERNALS__?: unknown}).__TAURI_INTERNALS__ ||
    window.location.protocol === 'tauri:');

const emptyConstraints: Constraints = {
  setting:'', pose:'', expression:'', character:'', dress:'', composition:'', additional:'',
  randomLoraMin:2, randomLoraMax:4,
};

const defaultSystemPrompt =
  'You are a Stable Diffusion prompt planner. Understand the purpose of every selected LoRA from its name, type, tags and description before writing the prompt. Use each LoRA only for concepts it plausibly provides. The character LoRA is the sole authority for character identity, appearance and named-character traits. Supporting LoRAs can contribute only their documented visual concept, style or content. Do not invent unsupported LoRA effects. Do not output LoRA activation words, angle-bracket LoRA syntax, weights, or implementation details. The application will append activation triggers deterministically. Return JSON only with positive_prompt, negative_prompt, rationale.';

const defaultDemographicPrompts: DemographicPrompts = {
  safe:'Keep all generated content non-sexual, non-explicit and suitable for general audiences. Avoid nudity and sexualized framing.',
  suggestive:'Allow mature, flirtatious or suggestive styling, but do not generate explicit sexual acts or graphic sexual detail.',
  explicit:'Allow adult sexual content between consenting adults, including nudity and explicit sexual detail, while avoiding minors or non-consensual scenarios.',
  'no-limits':'Do not add additional content restrictions beyond the application, model, platform and system requirements already in force.',
};

const stages: Array<{key: Stage; label: string}> = [
  {key:'library',label:'LIBRARY'},
  {key:'compatibility',label:'COMPATIBILITY'},
  {key:'selection',label:'MODEL STACK'},
  {key:'llm',label:'LLM PROMPT'},
  {key:'workflow',label:'WORKFLOW'},
  {key:'comfy',label:'COMFYUI'},
  {key:'recorded',label:'RECORDED'},
];

async function apiInvoke<T>(command:string, args:Record<string, unknown> = {}):Promise<T>{
  if(isTauriRuntime){
    return invoke<T>(command, args);
  }
  const response = await fetch('/api/' + command, {
    method:'POST',
    headers:{'content-type':'application/json'},
    body:JSON.stringify(args),
  });
  const text = await response.text();
  let value:unknown = null;
  try { value = text ? JSON.parse(text) : null; } catch { value = text; }
  if(!response.ok){
    const message = typeof value === 'object' && value && 'error' in value
      ? String((value as {error:unknown}).error)
      : text || response.statusText;
    throw new Error(message);
  }
  return value as T;
}

async function consumeSse(
  url:string,
  body:Record<string, unknown>,
  onData:(value:any)=>void,
){
  const response = await fetch(url, {
    method:'POST',
    headers:{'content-type':'application/json'},
    body:JSON.stringify(body),
  });
  if(!response.ok) throw new Error((await response.text()) || response.statusText);
  if(!response.body) throw new Error('Streaming response body is unavailable.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while(true){
    const {value, done} = await reader.read();
    if(done) break;
    buffer += decoder.decode(value, {stream:true});

    while(true){
      const split = buffer.indexOf('\n\n');
      if(split < 0) break;
      const packet = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const data = packet
        .split('\n')
        .filter(line => line.startsWith('data:'))
        .map(line => line.slice(5).trim())
        .join('\n');
      if(!data) continue;

      const value = JSON.parse(data);
      onData(value);
      if(value.error) throw new Error(String(value.error));
      if(value.done) return;
    }
  }
}

async function streamLlm(
  req: {settings:LlmSettings; systemPrompt:string; userPrompt:string},
  onText:(text:string)=>void,
){
  if(isTauriRuntime){
    const channel = new Channel<{text:string}>();
    channel.onmessage = event => onText(event.text);
    await invoke('stream_llm', {req, onEvent:channel});
    return;
  }
  await consumeSse('/api/stream_llm', {req}, value => {
    if(value.text) onText(String(value.text));
  });
}

async function monitorComfy(
  req:{comfyUrl:string; promptId:string},
  onProgress:(event:{percent:number; current:number; total:number; node:string|null; status:string})=>void,
){
  if(isTauriRuntime){
    const channel = new Channel<{
      percent:number; current:number; total:number; node:string|null; status:string;
    }>();
    channel.onmessage = onProgress;
    return invoke<{imageDataUrl:string|null; filename:string|null}>('monitor_comfy_generation', {
      req,
      onEvent:channel,
    });
  }

  let result:{imageDataUrl:string|null; filename:string|null}|null = null;
  await consumeSse('/api/monitor_comfy_generation', {req}, value => {
    if(value.progress) onProgress(value.progress);
    if(value.result) result = value.result;
  });
  if(!result) throw new Error('ComfyUI monitor ended without an image result.');
  return result;
}

function normUi(value:string){
  return value.trim().toLowerCase().replace(/[ _\-./]+/g,'');
}

function isCompatibleLoraUi(
  lora:LibrarySnapshot['loras'][number],
  checkpoint:LibrarySnapshot['checkpoints'][number],
){
  const keys=[...checkpoint.tags,checkpoint.baseModel || '',checkpoint.name]
    .map(normUi).filter(x=>x.length>2);
  const explicit=!!lora.baseModel && keys.some(k=>{
    const b=normUi(lora.baseModel || '');
    return k===b || k.includes(b) || b.includes(k);
  });
  const tagged=lora.tags.some(tag=>{
    const t=normUi(tag);
    return keys.some(k=>k===t || k.includes(t) || t.includes(k));
  });
  return explicit || tagged;
}

function App(){
  const [provider,setProvider]=useState<ProviderKind>('ollama');
  const [llm,setLlm]=useState<LlmSettings>({
    provider:'ollama',
    baseUrl:'http://127.0.0.1:11434',
    apiKey:'',
    model:'',
    temperature:0.75,
    maxTokens:1200,
  });
  const [models,setModels]=useState<string[]>([]);
  const [library,setLibrary]=useState<LibrarySnapshot|null>(null);
  const [selectedId,setSelectedId]=useState('');
  const [comfyRoot,setComfyRoot]=useState('D:\\ComfyUI\\models');
  const [raphaelRoot,setRaphaelRoot]=useState('');
  const [comfyUrl,setComfyUrl]=useState('http://127.0.0.1:8188');

  const [constraints,setConstraints]=useState<Constraints>(emptyConstraints);
  const [systemPrompt,setSystemPrompt]=useState(defaultSystemPrompt);
  const [demographic,setDemographic]=useState<DemographicLevel>('safe');
  const [demographicPrompts,setDemographicPrompts]=useState<DemographicPrompts>(defaultDemographicPrompts);
  const [maxLoras,setMaxLoras]=useState(4);
  const [width,setWidth]=useState(1024);
  const [height,setHeight]=useState(1024);
  const [steps,setSteps]=useState(28);
  const [cfg,setCfg]=useState(6.5);
  const [sampler,setSampler]=useState('euler');

  const [prepared,setPrepared]=useState<PreparedGeneration|null>(null);
  const [prompts,setPrompts]=useState<PromptPair|null>(null);
  const [stream,setStream]=useState('');
  const [history,setHistory]=useState<GenerationRecord[]>([]);
  const [tab,setTab]=useState<'generate'|'history'>('generate');

  const [stage,setStage]=useState<Stage>('library');
  const [status,setStatus]=useState<Record<Stage,Status>>({
    library:'idle', compatibility:'idle', selection:'idle', llm:'idle',
    workflow:'idle', comfy:'idle', recorded:'idle',
  });
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const [toast,setToast]=useState('');
  const [thumbs,setThumbs]=useState<Record<string,string>>({});
  const [selectedLoraIds,setSelectedLoraIds]=useState<string[]>([]);

  const [generationSettingsOpen,setGenerationSettingsOpen]=useState(false);
  const [modelSettingsOpen,setModelSettingsOpen]=useState(false);
  const [generationDraft,setGenerationDraft]=useState({
    llm,
    systemPrompt,
    demographic,
    demographicPrompts,
    maxLoras,
    randomLoraMin:constraints.randomLoraMin,
    randomLoraMax:constraints.randomLoraMax,
    constraints,
    width,
    height,
    steps,
    cfg,
    sampler,
  });
  const [checkpointSearch,setCheckpointSearch]=useState('');
  const [loraSearch,setLoraSearch]=useState('');

  const [webHost,setWebHost]=useState<WebHostInfo|null>(null);
  const [webHostBusy,setWebHostBusy]=useState(false);
  const [webHostError,setWebHostError]=useState('');

  const [comfyProgress,setComfyProgress]=useState(0);
  const [comfyCurrentStep,setComfyCurrentStep]=useState(0);
  const [comfyTotalSteps,setComfyTotalSteps]=useState(0);
  const [comfyCurrentNode,setComfyCurrentNode]=useState('');
  const [comfyStatus,setComfyStatus]=useState('idle');
  const [resultImage,setResultImage]=useState('');
  const [resultFilename,setResultFilename]=useState('');

  const streamText=useRef('');

  const selected=useMemo(
    ()=>library?.checkpoints.find(x=>x.id===selectedId) || library?.checkpoints[0],
    [library,selectedId],
  );

  const compatibleLoras=useMemo(
    ()=>library && selected ? library.loras.filter(lora=>isCompatibleLoraUi(lora,selected)) : [],
    [library,selected],
  );

  const manualLoras=useMemo(
    ()=>compatibleLoras.filter(lora=>selectedLoraIds.includes(lora.id)),
    [compatibleLoras,selectedLoraIds],
  );

  const filteredCheckpoints=useMemo(()=>{
    const q=checkpointSearch.trim().toLowerCase();
    if(!q) return library?.checkpoints || [];
    return (library?.checkpoints || []).filter(model =>
      [model.name,model.baseModel || '',...model.tags].join(' ').toLowerCase().includes(q)
    );
  },[library,checkpointSearch]);

  const filteredLoras=useMemo(()=>{
    const q=loraSearch.trim().toLowerCase();
    return compatibleLoras.filter(lora =>
      !q || [lora.name,lora.baseModel || '',...lora.tags].join(' ').toLowerCase().includes(q)
    );
  },[compatibleLoras,loraSearch]);

  useEffect(()=>{
    void loadHistory();
    void refreshModels();
    void discoverRoots();
  },[]);

  useEffect(()=>{
    if(!library) return;
    for(const model of [...library.checkpoints,...library.loras]){
      if(!model.thumbnail || thumbs[model.id]) continue;
      void apiInvoke<string>('path_to_data_url',{path:model.thumbnail})
        .then(url=>setThumbs(x=>({...x,[model.id]:url})))
        .catch(()=>undefined);
    }
  },[library]);

  const setStageStatus=(key:Stage,value:Status)=>{
    setStatus(x=>({...x,[key]:value}));
  };

  async function discoverRoots(){
    try{
      const config=await apiInvoke<{models_root:string|null;db_path:string|null}>('discover_raphael_config');
      if(config.models_root) setComfyRoot(config.models_root);
      if(config.db_path) setRaphaelRoot(config.db_path);
      if(config.models_root) await scan(config.models_root,config.db_path || undefined);
    }catch{}
  }

  async function loadHistory(){
    try{
      const records=await apiInvoke<Array<{payload:GenerationRecord}>>('load_history');
      setHistory(records.map(x=>x.payload));
    }catch{}
  }

  async function refreshModels(){
    try{
      const found=await apiInvoke<string[]>('list_provider_models',{settings:llm});
      setModels(found);
      if(!llm.model && found[0]) setLlm(x=>({...x,model:found[0]}));
    }catch(e){
      setError(String(e));
    }
  }

  async function scan(rootOverride=comfyRoot,raphaelOverride=raphaelRoot || undefined){
    setError('');
    setStage('library');
    setStageStatus('library','running');
    try{
      const snap=await apiInvoke<LibrarySnapshot>('scan_library',{
        req:{comfyRoot:rootOverride,raphaelRoot:raphaelOverride || null},
      });
      setLibrary(snap);
      if(!selectedId && snap.checkpoints[0]) setSelectedId(snap.checkpoints[0].id);
      const ids=new Set(snap.loras.map(x=>x.id));
      setSelectedLoraIds(current=>current.filter(id=>ids.has(id)));
      setStageStatus('library','done');
      setToast(snap.checkpoints.length + ' checkpoints · ' + snap.loras.length + ' LoRAs');
    }catch(e){
      setStageStatus('library','error');
      setError(String(e));
    }
  }

  function updateConstraint<K extends keyof Constraints>(key:K,value:Constraints[K]){
    setConstraints(x=>({...x,[key]:value}));
  }

  function openGenerationSettings(){
    setGenerationDraft({
      llm:{...llm},
      systemPrompt,
      demographic,
      demographicPrompts:{...demographicPrompts},
      maxLoras,
      randomLoraMin:constraints.randomLoraMin,
      randomLoraMax:Math.min(maxLoras,constraints.randomLoraMax),
      constraints:{...constraints},
      width,
      height,
      steps,
      cfg,
      sampler,
    });
    setGenerationSettingsOpen(true);
  }

  function saveGenerationSettings(){
    const draft=generationDraft;
    const cappedMax=Math.max(1,Math.min(16,draft.maxLoras));
    const nextConstraints={
      ...draft.constraints,
      randomLoraMin:Math.max(1,Math.min(cappedMax,draft.randomLoraMin)),
      randomLoraMax:Math.max(1,Math.min(cappedMax,draft.randomLoraMax)),
    };
    setProvider(draft.llm.provider);
    setLlm({...draft.llm,provider:draft.llm.provider});
    setConstraints(nextConstraints);
    setSystemPrompt(draft.systemPrompt);
    setDemographic(draft.demographic);
    setDemographicPrompts({...draft.demographicPrompts});
    setMaxLoras(cappedMax);
    setWidth(Math.max(64,draft.width));
    setHeight(Math.max(64,draft.height));
    setSteps(Math.max(1,draft.steps));
    setCfg(Math.max(0,draft.cfg));
    setSampler(draft.sampler || 'euler');
    setGenerationSettingsOpen(false);
    setToast('Generation settings saved');
  }

  async function rollStack(){
    if(!selected || !library){
      setError('Open Model Settings and select a checkpoint first.');
      return null;
    }
    setError('');
    setStage('compatibility');
    setStageStatus('compatibility','running');
    try{
      const result=await apiInvoke<PreparedGeneration>('prepare_generation',{
        req:{
          checkpoint:selected,
          loras:library.loras,
          selectedLoraIds:[],
          ...constraints,
          randomLoraMax:Math.min(maxLoras,constraints.randomLoraMax),
        },
      });
      setPrepared(result);
      setSelectedLoraIds(result.loras.map(lora=>lora.id));
      setStageStatus('compatibility','done');
      setStage('selection');
      setStageStatus('selection','done');
      return result;
    }catch(e){
      setStageStatus('compatibility','error');
      setStageStatus('selection','error');
      setError(String(e));
      return null;
    }
  }

  function toggleLora(id:string){
    setSelectedLoraIds(current=>current.includes(id)
      ? current.filter(x=>x!==id)
      : [...current,id]);
    setPrepared(null);
    setPrompts(null);
  }

  async function generate(){
    if(busy || !selected || !library) return;
    setBusy(true);
    setError('');
    setToast('');
    setComfyProgress(0);
    setComfyCurrentStep(0);
    setComfyTotalSteps(0);
    setComfyCurrentNode('');
    setComfyStatus('idle');
    setResultImage('');
    setResultFilename('');

    try{
      const generationSelectedLoraIds=selectedLoraIds.filter(id=>compatibleLoras.some(lora=>lora.id===id));

      setStage('compatibility');
      setStageStatus('compatibility','running');
      const prep=await apiInvoke<PreparedGeneration>('prepare_generation',{
        req:{
          checkpoint:selected,
          loras:library.loras,
          selectedLoraIds:generationSelectedLoraIds,
          ...constraints,
          randomLoraMax:Math.min(maxLoras,constraints.randomLoraMax),
        },
      });
      setPrepared(prep);
      setStageStatus('compatibility','done');
      setStage('selection');
      setStageStatus('selection','done');

      setStage('llm');
      setStageStatus('llm','running');
      streamText.current='';
      setStream('');
      setPrompts(null);

      const loraMetadata=prep.loras.map((l,index)=>
        'LORA ' + (index+1) + '\n' +
        'NAME: ' + l.name + '\n' +
        'TYPE: ' + (l.character ? 'CHARACTER IDENTITY' : 'SUPPORTING CONCEPT') + '\n' +
        'BASE MODEL: ' + (l.baseModel || 'unknown') + '\n' +
        'TAGS: ' + (l.tags.length ? l.tags.join(', ') : '(none)') + '\n' +
        'DESCRIPTION: ' + (l.description || '(none)')
      ).join('\n\n');

      const combinedSystemPrompt=[
        systemPrompt,
        'DEMOGRAPHIC POLICY (' + demographic.toUpperCase() + '):\n' + demographicPrompts[demographic],
      ].filter(Boolean).join('\n\n');

      const userPrompt=
        'CHECKPOINT: ' + prep.checkpoint.name + '\n' +
        'BASE: ' + (prep.checkpoint.baseModel || 'unknown') + '\n' +
        'COMPATIBILITY: ' + prep.compatibilityKeys.join(', ') + '\n\n' +
        'SELECTED LoRAs AND THEIR DOCUMENTED PURPOSE METADATA:\n' + loraMetadata + '\n\n' +
        'SCENE:\n' +
        'CHARACTER: ' + prep.scene.character + '\n' +
        'SETTING: ' + prep.scene.setting + '\n' +
        'POSE: ' + prep.scene.pose + '\n' +
        'EXPRESSION: ' + prep.scene.expression + '\n' +
        'DRESS: ' + prep.scene.dress + '\n' +
        'COMPOSITION: ' + prep.scene.composition + '\n' +
        'EXTRA: ' + (constraints.additional || '(none)') + '\n\n' +
        'Write a complete, coherent positive prompt that uses the selected LoRAs according to their documented purposes, plus a robust negative prompt.';

      await streamLlm({
        settings:llm,
        systemPrompt:combinedSystemPrompt,
        userPrompt,
      },event=>{
        streamText.current+=event;
        flushSync(()=>setStream(streamText.current));
      });

      const rawPair=await apiInvoke<PromptPair>('parse_prompt_pair',{raw:streamText.current});
      const pair=await apiInvoke<PromptPair>('finalize_prompt_pair',{
        req:{promptPair:rawPair,loras:prep.loras},
      });
      setPrompts(pair);
      setStageStatus('llm','done');

      setStage('workflow');
      setStageStatus('workflow','running');
      const workflow=await apiInvoke<Record<string,unknown>>('build_workflow',{
        req:{
          checkpoint:prep.checkpoint,
          loras:prep.loras,
          width,
          height,
          steps,
          cfg,
          sampler,
          seed:Math.floor(Math.random()*2147000000),
        },
      });
      const injected=await apiInvoke<Record<string,unknown>>('inject_prompts',{
        req:{
          workflow,
          positivePrompt:pair.positive_prompt,
          negativePrompt:pair.negative_prompt,
        },
      });
      setStageStatus('workflow','done');

      setStage('comfy');
      setStageStatus('comfy','running');
      setComfyStatus('submitting');
      let promptId:string|undefined;
      try{
        const response=await apiInvoke<{prompt_id?:string}>('submit_to_comfy',{
          req:{comfyUrl,workflow:injected},
        });
        promptId=response.prompt_id;
        if(!promptId) throw new Error('ComfyUI did not return a prompt_id.');

        setComfyStatus('waiting');
        const generation=await monitorComfy({comfyUrl,promptId},event=>{
          setComfyProgress(event.percent);
          setComfyCurrentStep(event.current);
          setComfyTotalSteps(event.total);
          setComfyCurrentNode(event.node || '');
          setComfyStatus(event.status);
        });

        if(generation.imageDataUrl) setResultImage(generation.imageDataUrl);
        if(generation.filename) setResultFilename(generation.filename);
        setComfyProgress(100);
        setComfyStatus('done');
        setStageStatus('comfy','done');
      }catch(e){
        setComfyStatus('error');
        setStageStatus('comfy','error');
        setError('ComfyUI generation failed: ' + String(e));
      }

      const record:GenerationRecord={
        id:crypto.randomUUID(),
        timestamp:new Date().toISOString(),
        provider:llm.provider,
        model:llm.model,
        checkpoint:prep.checkpoint,
        loras:prep.loras,
        scene:prep.scene,
        positivePrompt:pair.positive_prompt,
        negativePrompt:pair.negative_prompt,
        workflow:injected,
        comfyPromptId:promptId,
      };
      await apiInvoke('append_history',{payload:record});
      setHistory(x=>[record,...x].slice(0,100));
      setStage('recorded');
      setStageStatus('recorded','done');
      setToast(promptId ? 'Generation complete · ' + promptId : 'Generation recorded');
    }catch(e){
      setError(String(e));
      setStageStatus(stage,'error');
    }finally{
      setBusy(false);
    }
  }

  async function pickFolder(setter:(v:string)=>void){
    if(!isTauriRuntime){
      setError('Folder browsing is available on the desktop host. Enter the path manually from the LAN client.');
      return;
    }
    const picked=await apiInvoke<{path?:string|null}>('pick_folder');
    if(picked.path) setter(picked.path);
  }

  async function copy(text:string){
    await navigator.clipboard?.writeText(text);
    setToast('Copied');
  }

  async function startLanHost(){
    if(!isTauriRuntime) return;
    setWebHostBusy(true);
    setWebHostError('');
    try{
      const info=await apiInvoke<WebHostInfo>('start_web_host',{port:1421});
      setWebHost(info);
      setToast('LAN host ready · ' + info.lanUrl);
    }catch(e){
      setWebHostError(String(e));
    }finally{
      setWebHostBusy(false);
    }
  }

  async function stopLanHost(){
    if(!isTauriRuntime) return;
    setWebHostBusy(true);
    try{
      await apiInvoke('stop_web_host');
      setWebHost(null);
      setToast('LAN host stopped');
    }catch(e){
      setWebHostError(String(e));
    }finally{
      setWebHostBusy(false);
    }
  }

  return <div className="app-shell">
    <iframe className="raphael-bg" src="/raphael-background.html" title="Raphael background" aria-hidden="true"/>
    <div className="vignette"/>

    <header className="topbar">
      <div className="brand">
        <div className="brand-mark"><Sparkles size={15}/></div>
        <div>
          <div className="brand-title">PROMPT FORGE</div>
          <div className="brand-sub">RAPHAEL GENERATION ENGINE</div>
        </div>
      </div>

      <div className="top-actions">
        <button onClick={openGenerationSettings}><Settings2 size={13}/> GENERATION SETTINGS</button>
        <button onClick={()=>setModelSettingsOpen(true)}><Layers3 size={13}/> MODEL SETTINGS</button>
        {isTauriRuntime && (
          <button className={webHost ? 'host-active' : ''} onClick={()=>void (webHost ? stopLanHost() : startLanHost())}>
            <Globe2 size={13}/> {webHost ? 'LAN ONLINE' : 'LAN WEB HOST'}
          </button>
        )}
        <span className="top-status"><span className="pulse-dot"/>{busy ? 'GENERATING' : 'READY'}</span>
      </div>
    </header>

    <aside className="left-rail">
      <div className="rail-label">WORKSPACE</div>
      <button className={'rail-btn ' + (tab==='generate' ? 'active' : '')} onClick={()=>setTab('generate')}><WandSparkles size={15}/> GENERATE</button>
      <button className={'rail-btn ' + (tab==='history' ? 'active' : '')} onClick={()=>setTab('history')}><History size={15}/> HISTORY <span>{history.length}</span></button>

      <div className="rail-spacer"/>
      <div className="root-box">
        <div className="root-label">COMFYUI ROOT</div>
        <div className="root-path">{comfyRoot}</div>
        <button onClick={()=>void pickFolder(setComfyRoot)} disabled={!isTauriRuntime}><FolderOpen size={13}/> BROWSE</button>
      </div>

      {webHost && <div className="host-box">
        <div className="root-label">LAN WEB HOST</div>
        <div className="host-url">{webHost.lanUrl}</div>
        <button onClick={()=>void copy(webHost.lanUrl)}><Copy size={12}/> COPY ADDRESS</button>
      </div>}
    </aside>

    <main className="content">
      <section className="stage-strip">
        {stages.map((s,i)=><div className={'stage ' + status[s.key] + ' ' + (stage===s.key ? 'current' : '')} key={s.key}>
          <div className="stage-index">{status[s.key]==='done' ? <Check size={12}/> : status[s.key]==='error' ? <CircleAlert size={12}/> : i+1}</div>
          <div className="stage-label">{s.label}</div>
        </div>)}
      </section>

      {tab==='generate' && <>
        <section className="config-strip">
          <div className="config-card">
            <div className="config-card-head">
              <div><div className="kicker">ACTIVE MODEL</div><div className="config-title">{selected?.name || 'NO CHECKPOINT SELECTED'}</div></div>
              <button onClick={()=>setModelSettingsOpen(true)}><Layers3 size={12}/> EDIT MODELS</button>
            </div>
            <div className="config-meta">{selected?.baseModel || 'BASE UNKNOWN'} · {manualLoras.length} compatible LoRAs selected · {selectedLoraIds.length} saved in session</div>
          </div>

          <div className="config-card">
            <div className="config-card-head">
              <div><div className="kicker">GENERATION PROFILE</div><div className="config-title">{demographic.toUpperCase()} · {llm.model || 'NO LLM MODEL'}</div></div>
              <button onClick={openGenerationSettings}><Settings2 size={12}/> EDIT GENERATION</button>
            </div>
            <div className="config-meta">
              {constraints.setting || 'random setting'} · {constraints.pose || 'random pose'} · {constraints.expression || 'random expression'} · max {maxLoras} LoRAs
            </div>
          </div>
        </section>

        <section className="lower-grid">
          <div className="panel prompt-panel">
            <div className="panel-head">
              <div><div className="kicker">STREAMED MODEL OUTPUT</div><div className="panel-title">PROMPT ENGINE</div></div>
              <span className="provider-chip">{provider==='ollama' ? 'OLLAMA' : 'OPENAI COMPAT'}</span>
            </div>
            <div className="stream-box">{stream
              ? <pre>{stream}</pre>
              : <div className="stream-placeholder"><Terminal size={18}/><span>LLM output will stream here token by token.</span></div>}
            </div>

            {prompts && <div className="prompt-result">
              <div className="prompt-block">
                <div className="prompt-block-head"><span>POSITIVE</span><button onClick={()=>void copy(prompts.positive_prompt)}><Copy size={12}/> COPY</button></div>
                <div className="prompt-text">{prompts.positive_prompt}</div>
              </div>
              <div className="prompt-block">
                <div className="prompt-block-head"><span>NEGATIVE</span><button onClick={()=>void copy(prompts.negative_prompt)}><Copy size={12}/> COPY</button></div>
                <div className="prompt-text">{prompts.negative_prompt}</div>
              </div>
            </div>}

            {prepared && <div className="stack-preview">
              <div className="stack-head"><span>ACTIVE LoRA STACK</span><span>{prepared.loras.length} / {maxLoras}</span></div>
              {prepared.loras.map(l=><div className="stack-item" key={l.id}>
                <span className={'stack-dot ' + (l.character ? 'character' : '')}/>
                <div><b>{l.name}</b><small>{l.activationTags.join(', ') || 'no activation metadata'}</small></div>
                <strong>{l.weight.toFixed(2)}</strong>
              </div>)}
            </div>}

            <div className="run-row">
              <button className="secondary-btn" disabled={busy || !selected} onClick={()=>void rollStack()}><RefreshCw size={14}/> RANDOMIZE LoRAs</button>
              <button className="primary-btn" disabled={busy || !selected || !llm.model} onClick={()=>void generate()}><Play size={15}/> {busy ? 'GENERATING…' : 'GENERATE'}</button>
            </div>
          </div>
        </section>

        <section className="panel generation-result-panel">
          <div className="panel-head">
            <div><div className="kicker">COMFYUI GENERATION MONITOR</div><div className="panel-title">IMAGE OUTPUT</div></div>
            <span className="provider-chip">{comfyStatus==='done' ? 'COMPLETE' : comfyStatus==='error' ? 'ERROR' : comfyStatus==='idle' ? 'IDLE' : comfyStatus.toUpperCase()}</span>
          </div>
          <div className="comfy-progress-wrap">
            <div className="comfy-progress-meta">
              <span>{comfyCurrentStep && comfyTotalSteps ? 'STEP ' + comfyCurrentStep + ' / ' + comfyTotalSteps : comfyStatus==='done' ? 'COMPLETE' : 'WAITING FOR COMFYUI'}</span>
              <strong>{Math.round(comfyProgress)}%</strong>
            </div>
            <div className="comfy-progress-track"><div className="comfy-progress-fill" style={{width:comfyProgress + '%'}}/></div>
            {comfyCurrentNode && <div className="comfy-node">NODE {comfyCurrentNode}</div>}
          </div>

          {resultImage ? <div className="result-image-wrap">
            <img className="result-image" src={resultImage} alt={resultFilename || 'Generated result'}/>
            {resultFilename && <div className="result-filename">{resultFilename}</div>}
          </div> :
          <div className="result-placeholder"><WandSparkles size={22}/><span>{comfyStatus==='error' ? 'Generation failed. See the error message below.' : 'Your generated image will appear here when ComfyUI finishes.'}</span></div>}
        </section>
      </>}

      {tab==='history' && <section className="history-panel">
        <div className="section-head">
          <div><div className="kicker">LOCAL RUN ARCHIVE</div><div className="section-title">GENERATION HISTORY</div></div>
          <button onClick={()=>void loadHistory()}><RefreshCw size={13}/> REFRESH</button>
        </div>
        {!history.length
          ? <div className="empty-state"><History size={22}/><div><b>NO GENERATIONS RECORDED</b><span>Prompt, LoRA, workflow and ComfyUI records appear here.</span></div></div>
          : <div className="history-list">{history.map(item=><article className="history-card" key={item.id}>
            <div className="history-main">
              <div className="history-meta"><span>{new Date(item.timestamp).toLocaleString()}</span><span>{item.model || '—'}</span><span>{item.checkpoint.name}</span></div>
              <div className="history-scene">{item.scene.character} · {item.scene.setting} · {item.scene.pose} · {item.scene.expression}</div>
              <div className="history-loras">{item.loras.map(l=><span key={l.id}>{l.name}</span>)}</div>
            </div>
            <button className="icon-btn" onClick={()=>void copy(item.positivePrompt)}><Copy size={14}/></button>
          </article>)}</div>}
      </section>}

      {error && <div className="error-box"><CircleAlert size={14}/><span>{error}</span></div>}
      {toast && <div className="toast">{toast}</div>}
      {webHostError && <div className="error-box"><CircleAlert size={14}/><span>{webHostError}</span></div>}
    </main>

    <aside className="right-rail">
      <div className="rail-label">ACTIVE CHECKPOINT</div>
      {selected ? <div className="active-model">
        <div className="active-model-thumb">{thumbs[selected.id] ? <img src={thumbs[selected.id]} alt=""/> : <Layers3 size={24}/>}</div>
        <div className="active-name">{selected.name}</div>
        <div className="active-sub">{selected.baseModel || 'BASE NOT RESOLVED'}</div>
        <div className="active-tags">{selected.tags.slice(0,7).map(t=><span key={t}>{t}</span>)}</div>
      </div> : <div className="active-empty">SELECT A CHECKPOINT</div>}
      <div className="rail-divider"/>
      <div className="rail-label">SESSION LoRAs</div>
      <div className="active-tags">{selectedLoraIds.length ? selectedLoraIds.slice(0,8).map(id=><span key={id}>{library?.loras.find(l=>l.id===id)?.name || id}</span>) : <span>NONE</span>}</div>
      <div className="rail-divider"/><div className="rail-label">LIVE STATUS</div>
      <div className="stage-stack">{stages.map(s=><div className="mini-stage" key={s.key}><span className={'status ' + status[s.key]}/><span>{s.label}</span><b>{status[s.key].toUpperCase()}</b></div>)}</div>
    </aside>

    {generationSettingsOpen && <div className="overlay">
      <div className="modal generation-modal">
        <div className="modal-head">
          <div><div className="kicker">CONFIGURATION / LLM + SCENE</div><div className="section-title">GENERATION SETTINGS</div></div>
          <button className="icon-btn" onClick={()=>setGenerationSettingsOpen(false)}><X size={16}/></button>
        </div>

        <div className="modal-scroll">
          <section className="settings-section">
            <div className="settings-section-title">MODEL PROVIDER</div>
            <div className="provider-toggle">
              <button className={generationDraft.llm.provider==='ollama' ? 'active' : ''} onClick={()=>setGenerationDraft(d=>({...d,llm:{...d.llm,provider:'ollama',baseUrl:'http://127.0.0.1:11434'}}))}>OLLAMA</button>
              <button className={generationDraft.llm.provider==='openai-compatible' ? 'active' : ''} onClick={()=>setGenerationDraft(d=>({...d,llm:{...d.llm,provider:'openai-compatible',baseUrl:'http://127.0.0.1:8080/v1'}}))}>OPENAI COMPATIBLE</button>
            </div>
            <div className="field-grid">
              <label className="wide-field"><span>BASE URL</span><input value={generationDraft.llm.baseUrl} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,baseUrl:e.target.value}}))}/></label>
              <label className="wide-field"><span>API KEY / OPTIONAL</span><input type="password" value={generationDraft.llm.apiKey} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,apiKey:e.target.value}}))}/></label>
              <label className="wide-field"><span>MODEL</span>
                <select value={generationDraft.llm.model} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,model:e.target.value}}))}>
                  <option value="">SELECT MODEL…</option>{models.map(model=><option key={model}>{model}</option>)}
                </select>
              </label>
              <div className="inline-controls">
                <button className="secondary-btn" onClick={()=>void (async()=>{const before=llm;try{setLlm(generationDraft.llm);await refreshModels();}finally{setLlm(before);}})()}><RefreshCw size={13}/> GET MODELS</button>
              </div>
              <label className="wide-field"><span>TEMPERATURE · {generationDraft.llm.temperature.toFixed(2)}</span><input type="range" min={0} max={2} step={0.05} value={generationDraft.llm.temperature} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,temperature:Number(e.target.value)}}))}/></label>
              <label className="wide-field"><span>MAX OUTPUT TOKENS</span><input type="number" min={128} max={16384} value={generationDraft.llm.maxTokens} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,maxTokens:Math.max(128,Number(e.target.value))}}))}/></label>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">SYSTEM PROMPT</div>
            <textarea className="settings-textarea tall" value={generationDraft.systemPrompt} onChange={e=>setGenerationDraft(d=>({...d,systemPrompt:e.target.value}))}/>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">DEMOGRAPHIC POLICY</div>
            <div className="demographic-grid">
              {(['safe','suggestive','explicit','no-limits'] as DemographicLevel[]).map(level=>
                <button key={level} className={'demographic-card ' + (generationDraft.demographic===level ? 'active' : '')} onClick={()=>setGenerationDraft(d=>({...d,demographic:level}))}>
                  <b>{level.replace('-',' ').toUpperCase()}</b><span>{level==='safe' ? 'General audience' : level==='suggestive' ? 'Mature / suggestive' : level==='explicit' ? 'Adult explicit' : 'No added restriction'}</span>
                </button>
              )}
            </div>
            <div className="policy-editor-list">
              {(['safe','suggestive','explicit','no-limits'] as DemographicLevel[]).map(level=>
                <label className="wide-field" key={level}>
                  <span>{level.replace('-',' ').toUpperCase()} SYSTEM PROMPT</span>
                  <textarea className="settings-textarea" value={generationDraft.demographicPrompts[level]} onChange={e=>setGenerationDraft(d=>({...d,demographicPrompts:{...d.demographicPrompts,[level]:e.target.value}}))}/>
                </label>
              )}
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">SCENE + LoRA LIMITS</div>
            <div className="field-grid">
              {(['setting','pose','expression','character','dress','composition','additional'] as const).map(key=>
                <label className={'wide-field ' + (key==='additional' ? 'full-width' : '')} key={key}>
                  <span>{key.replace('_',' ').toUpperCase()}</span>
                  {key==='additional'
                    ? <textarea className="settings-textarea" value={generationDraft.constraints[key]} onChange={e=>setGenerationDraft(d=>({...d,constraints:{...d.constraints,[key]:e.target.value}}))} placeholder="Anything the prompt engine must obey…"/>
                    : <input value={generationDraft.constraints[key]} onChange={e=>setGenerationDraft(d=>({...d,constraints:{...d.constraints,[key]:e.target.value}}))} placeholder={key==='character' ? 'compatible character LoRA only' : 'blank = random'}/>}
                </label>
              )}
              <label className="wide-field"><span>MAX LoRAs</span><input type="number" min={1} max={16} value={generationDraft.maxLoras} onChange={e=>setGenerationDraft(d=>({...d,maxLoras:Math.max(1,Number(e.target.value))}))}/></label>
              <label className="wide-field"><span>RANDOM LoRA MIN</span><input type="number" min={1} max={16} value={generationDraft.randomLoraMin} onChange={e=>setGenerationDraft(d=>({...d,randomLoraMin:Math.max(1,Number(e.target.value))}))}/></label>
              <label className="wide-field"><span>RANDOM LoRA MAX</span><input type="number" min={1} max={16} value={generationDraft.randomLoraMax} onChange={e=>setGenerationDraft(d=>({...d,randomLoraMax:Math.max(1,Number(e.target.value))}))}/></label>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">SAMPLING</div>
            <div className="field-grid">
              <label className="wide-field"><span>WIDTH</span><input type="number" min={64} step={64} value={generationDraft.width} onChange={e=>setGenerationDraft(d=>({...d,width:Number(e.target.value)}))}/></label>
              <label className="wide-field"><span>HEIGHT</span><input type="number" min={64} step={64} value={generationDraft.height} onChange={e=>setGenerationDraft(d=>({...d,height:Number(e.target.value)}))}/></label>
              <label className="wide-field"><span>STEPS</span><input type="number" min={1} max={200} value={generationDraft.steps} onChange={e=>setGenerationDraft(d=>({...d,steps:Number(e.target.value)}))}/></label>
              <label className="wide-field"><span>CFG</span><input type="number" min={0} step={0.1} value={generationDraft.cfg} onChange={e=>setGenerationDraft(d=>({...d,cfg:Number(e.target.value)}))}/></label>
              <label className="wide-field full-width"><span>SAMPLER</span><input value={generationDraft.sampler} onChange={e=>setGenerationDraft(d=>({...d,sampler:e.target.value}))}/></label>
            </div>
          </section>
        </div>

        <div className="modal-foot">
          <button className="secondary-btn" onClick={()=>setGenerationSettingsOpen(false)}>CANCEL</button>
          <button className="primary-btn" onClick={saveGenerationSettings}>SAVE GENERATION SETTINGS</button>
        </div>
      </div>
    </div>}

    {modelSettingsOpen && <div className="overlay">
      <div className="modal model-modal">
        <div className="modal-head">
          <div><div className="kicker">LIBRARY / RAPHAEL CACHE</div><div className="section-title">MODEL SETTINGS</div></div>
          <button className="icon-btn" onClick={()=>setModelSettingsOpen(false)}><X size={16}/></button>
        </div>

        <div className="modal-scroll">
          <section className="settings-section">
            <div className="settings-section-title">GENERATION BACKEND</div>
            <div className="field-grid">
              <label className="wide-field"><span>COMFYUI API</span><input value={comfyUrl} onChange={e=>setComfyUrl(e.target.value)}/></label>
              <label className="wide-field"><span>COMFYUI MODELS ROOT</span><div className="input-button"><input value={comfyRoot} onChange={e=>setComfyRoot(e.target.value)}/><button onClick={()=>void pickFolder(setComfyRoot)} disabled={!isTauriRuntime}><FolderOpen size={13}/></button></div></label>
              <label className="wide-field"><span>RAPHAEL MANAGER DB / CACHE</span><div className="input-button"><input value={raphaelRoot} onChange={e=>setRaphaelRoot(e.target.value)}/><button onClick={()=>void pickFolder(setRaphaelRoot)} disabled={!isTauriRuntime}><FolderOpen size={13}/></button></div></label>
            </div>
            <button className="primary-btn full" onClick={()=>void scan()}><RefreshCw size={13}/> SCAN + MERGE CACHE</button>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">SELECT CHECKPOINT</div>
            <div className="search-row"><Search size={14}/><input value={checkpointSearch} onChange={e=>setCheckpointSearch(e.target.value)} placeholder="Search checkpoints, base models or tags…"/></div>
            <div className="model-modal-grid">
              {filteredCheckpoints.map(model=><button key={model.id} className={'modal-model-card ' + (selected?.id===model.id ? 'selected' : '')} onClick={()=>setSelectedId(model.id)}>
                <div className="modal-model-thumb">{thumbs[model.id] ? <img src={thumbs[model.id]} alt=""/> : <Layers3 size={26}/>}<span>{model.baseModel || 'UNKNOWN'}</span></div>
                <div className="model-body"><div className="model-name">{model.name}</div><div className="model-type">CHECKPOINT</div><div className="model-tags">{model.tags.slice(0,5).map(t=><span key={t}>{t}</span>)}</div></div>
              </button>)}
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">SELECT LoRAs · {selectedLoraIds.length} SAVED / {compatibleLoras.length} COMPATIBLE</div>
            <div className="search-row"><Search size={14}/><input value={loraSearch} onChange={e=>setLoraSearch(e.target.value)} placeholder="Search compatible LoRAs, tags or base model…"/></div>
            <div className="session-note">Selections are session-persistent. They remain selected until you explicitly unselect them or close the app.</div>
            <div className="model-modal-grid lora-modal-grid">
              {filteredLoras.map(lora=><button key={lora.id} className={'modal-model-card ' + (selectedLoraIds.includes(lora.id) ? 'selected' : '')} onClick={()=>toggleLora(lora.id)}>
                <div className="modal-model-thumb">{thumbs[lora.id] ? <img src={thumbs[lora.id]} alt=""/> : <Layers3 size={26}/>} {lora.character && <span>CHARACTER</span>}</div>
                <div className="model-body"><div className="model-name">{lora.name}</div><div className="model-type">{lora.character ? 'CHARACTER LoRA' : 'SUPPORT LoRA'}</div><div className="model-tags">{lora.tags.slice(0,6).map(t=><span key={t}>{t}</span>)}</div></div>
                <div className="modal-check">{selectedLoraIds.includes(lora.id) ? <Check size={13}/> : null}</div>
              </button>)}
            </div>
          </section>
        </div>

        <div className="modal-foot">
          <button className="secondary-btn" onClick={()=>setSelectedLoraIds([])}>UNSELECT ALL LoRAs</button>
          <button className="primary-btn" onClick={()=>setModelSettingsOpen(false)}>DONE</button>
        </div>
      </div>
    </div>}
  </div>;
}

export default App;
