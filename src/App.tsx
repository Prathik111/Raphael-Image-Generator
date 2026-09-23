import { useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { Channel, invoke } from '@tauri-apps/api/core';
import {
  Check, CircleAlert, Copy, Globe2, History, Layers3, Play,
  RefreshCw, Search, Settings2, Sparkles, Terminal, WandSparkles, X
} from 'lucide-react';
import type {
  Constraints, DemographicLevel, DemographicPrompts, GenerationRecord, GenerationSettings, LibrarySnapshot,
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
  'You are an expert Stable Diffusion prompt director and scene planner. Build the image prompt from the requested scene, not merely from LoRA metadata. Treat LoRAs as supporting visual tools, never as the source of the entire scene. The character LoRA is the sole authority for character identity, face, body identity and named-character traits. Supporting LoRAs may contribute only their documented concept, costume element, visual motif, style or other explicitly documented effect. Do not invent unsupported LoRA effects and do not let a LoRA replace explicit scene direction. ' +
  'You must explicitly decide and describe the subject state, action or activity, body pose, hand/arm placement, head direction, gaze, facial expression, emotional state, interaction with nearby objects, clothing/dress, setting, background/environment, time of day, weather or atmosphere when relevant, camera viewpoint, framing, shot type, perspective, depth, spatial arrangement, lighting direction and quality, color/mood, material/texture details, and important foreground/background elements. ' +
  'Do not rely on generic LoRA activation text to describe expression, state, pose, background or composition; those must be written as normal prompt language. Preserve requested scene constraints exactly when possible. Resolve conflicts by prioritizing explicit user scene constraints, then character identity, then compatible LoRA concepts. Keep the positive prompt coherent and ordered from subject/identity to action/state, appearance, pose/expression, clothing, environment/background, composition/camera, lighting and finishing details. Make the negative prompt targeted to the requested scene and common image-generation failures rather than adding random unrelated concepts. ' +
  'Never output LoRA activation words, angle-bracket LoRA syntax, weights, implementation details or JSON commentary outside the required object. Return JSON only with positive_prompt, negative_prompt, rationale.';

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
    temperature:0.72,
    maxTokens:4096,
    contextTokens:16384,
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
  const [manualLoraIds,setManualLoraIds]=useState<string[]>([]);

  const [settingsOpen,setSettingsOpen]=useState(false);
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
  const [selectedHistoryId,setSelectedHistoryId]=useState('');
  const [historySettingsVisible,setHistorySettingsVisible]=useState(false);
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
    ()=>compatibleLoras.filter(lora=>manualLoraIds.includes(lora.id)),
    [compatibleLoras,manualLoraIds],
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

  async function fetchModels(settings:LlmSettings){
    const found=await apiInvoke<string[]>('list_provider_models',{settings});
    setModels(found);
    return found;
  }

  async function refreshModels(){
    try{
      const found=await fetchModels(llm);
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
      setManualLoraIds(current=>current.filter(id=>ids.has(id)));
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
    setSettingsOpen(false);
    setToast('Generation settings saved');
  }

  async function rollStack(){
    if(!selected || !library){
      setError('Select a checkpoint first.');
      return null;
    }
    setError('');

    const manualIds=manualLoraIds.filter(id=>compatibleLoras.some(lora=>lora.id===id));
    const minCount=Math.max(1,Math.min(maxLoras,constraints.randomLoraMin));
    const maxCount=Math.max(minCount,Math.min(maxLoras,constraints.randomLoraMax));
    const randomTarget=Math.floor(Math.random()*(maxCount-minCount+1))+minCount;
    const targetCount=Math.max(manualIds.length,randomTarget);
    const availablePool=compatibleLoras
      .filter(lora=>!manualIds.includes(lora.id))
      .sort(()=>Math.random()-0.5);

    const randomSlots=Math.max(0,Math.min(maxLoras,targetCount)-manualIds.length);
    const randomIds=availablePool.slice(0,randomSlots).map(lora=>lora.id);
    const combinedIds=[...manualIds,...randomIds];

    setSelectedLoraIds(combinedIds);
    setPrepared(null);
    setPrompts(null);
    setStage('compatibility');
    setStageStatus('compatibility','running');

    try{
      const result=await apiInvoke<PreparedGeneration>('prepare_generation',{
        req:{
          checkpoint:selected,
          loras:library.loras,
          selectedLoraIds:combinedIds,
          ...constraints,
          randomLoraMax:Math.min(maxLoras,constraints.randomLoraMax),
        },
      });
      setPrepared(result);
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
    setManualLoraIds(current=>{
      const next=current.includes(id) ? current.filter(x=>x!==id) : [...current,id];
      setSelectedLoraIds(selected=>{
        const withoutId=selected.filter(x=>x!==id && compatibleLoras.some(lora=>lora.id===x));
        return next.includes(id) ? [...withoutId,id] : withoutId;
      });
      return next;
    });
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
    let generatedImageDataUrl='';
    let generatedImageFilename='';

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

        if(generation.imageDataUrl){
          generatedImageDataUrl=generation.imageDataUrl;
          setResultImage(generation.imageDataUrl);
        }
        if(generation.filename){
          generatedImageFilename=generation.filename;
          setResultFilename(generation.filename);
        }
        setComfyProgress(100);
        setComfyStatus('done');
        setStageStatus('comfy','done');
      }catch(e){
        setComfyStatus('error');
        setStageStatus('comfy','error');
        setError('ComfyUI generation failed: ' + String(e));
      }

      const historySettings:GenerationSettings={
        llm:{...llm,apiKey:llm.apiKey ? '••••••••' : ''},
        systemPrompt,
        demographic,
        demographicPrompts:{...demographicPrompts},
        maxLoras,
        randomLoraMin:constraints.randomLoraMin,
        randomLoraMax:constraints.randomLoraMax,
        constraints:{...constraints},
        width,
        height,
        steps,
        cfg,
        sampler,
      };
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
        rationale:pair.rationale,
        generationSettings:historySettings,
        imageDataUrl:generatedImageDataUrl || undefined,
        imageFilename:generatedImageFilename || undefined,
        workflow:injected,
        comfyPromptId:promptId,
      };
      await apiInvoke('append_history',{payload:record});
      setHistory(x=>[record,...x].slice(0,100));
      setSelectedHistoryId(record.id);
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

  function openHistory(id:string){
    setSelectedHistoryId(id);
    setHistorySettingsVisible(false);
    setTab('history');
    setSettingsOpen(false);
  }

  function selectCheckpoint(id:string){
    setSelectedId(id);
    setPrepared(null);
    setPrompts(null);
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
      const info=await apiInvoke<WebHostInfo>('start_web_host',{port:1424});
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

  const selectedHistory=selectedHistoryId ? history.find(item=>item.id===selectedHistoryId) : undefined;

  return <div className="app-shell">
    <iframe className="raphael-bg" src="/raphael-background.html" title="Raphael background" aria-hidden="true"/>
    <div className="vignette"/>

    <header className="topbar">
      <div className="brand">
        <div className="brand-mark"><Sparkles size={15}/></div>
        <div className="brand-title">PROMPT FORGE</div>
      </div>

      <div className="top-actions">
        {isTauriRuntime && (
          <button className={webHost ? 'host-active' : ''} onClick={()=>void (webHost ? stopLanHost() : startLanHost())}>
            <Globe2 size={13}/> {webHost ? 'WEB HOST ON' : 'WEB HOST'}
          </button>
        )}
        {(busy || comfyStatus==='done' || comfyStatus==='error') && (
          <span className={'top-status ' + (comfyStatus==='error' ? 'error' : '')}>
            <span className="pulse-dot"/>
            {busy ? 'GENERATING' : comfyStatus==='done' ? 'COMPLETE' : 'ERROR'}
          </span>
        )}
      </div>
    </header>

    <aside className="left-rail">
      <div className="rail-label">WORKSPACE</div>
      <button className={'rail-btn ' + (tab==='generate' ? 'active' : '')} onClick={()=>{setTab('generate');setSettingsOpen(false)}}><WandSparkles size={15}/> GENERATE</button>
      <button className={'rail-btn ' + (tab==='history' ? 'active' : '')} onClick={()=>{setTab('history');setSettingsOpen(false)}}><History size={15}/> HISTORY <span>{history.length}</span></button>
      <button className={'rail-btn ' + (settingsOpen ? 'active' : '')} onClick={()=>setSettingsOpen(v=>!v)}><Settings2 size={15}/> SETTINGS</button>

      <div className="recent-heading">RECENT</div>
      <div className="recent-images">
        {history.filter(item=>item.imageDataUrl).slice(0,8).map(item=>
          <button className="recent-image" key={item.id} onClick={()=>openHistory(item.id)} title={new Date(item.timestamp).toLocaleString()}>
            <img src={item.imageDataUrl} alt=""/>
          </button>
        )}
        {!history.some(item=>item.imageDataUrl) && <div className="recent-empty">NO IMAGES</div>}
      </div>

      {webHost && <div className="host-box">
        <div className="root-label">WEB HOST</div>
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
        <section className="workspace-panel">
          <div className="panel-head">
            <div>
              <div className="kicker">PROMPT ENGINE</div>
              <div className="panel-title">GENERATION</div>
            </div>
            <div className="workspace-meta">
              <span>{llm.model || 'NO LLM MODEL'}</span>
              <span>{selected?.name || 'NO CHECKPOINT'}</span>
              <span>{selectedLoraIds.length} LoRAs</span>
            </div>
          </div>

          <div className="stream-box">{stream
            ? <pre>{stream}</pre>
            : <div className="stream-placeholder"><Terminal size={18}/><span>Prompt output appears here.</span></div>}
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
            <div className="stack-head"><span>ACTIVE LORA STACK</span><span>{prepared.loras.length} / {maxLoras}</span></div>
            {prepared.loras.map(l=><div className="stack-item" key={l.id}>
              <span className={'stack-dot ' + (l.character ? 'character' : '')}/>
              <div><b>{l.name}</b><small>{l.activationTags.join(', ') || 'NO ACTIVATION METADATA'}</small></div>
              <strong>{l.weight.toFixed(2)}</strong>
            </div>)}
          </div>}

          <div className="run-row">
            <button className="secondary-btn" disabled={busy || !selected} onClick={()=>void rollStack()}><RefreshCw size={14}/> RANDOMIZE LORAS</button>
            <button className="primary-btn" disabled={busy || !selected || !llm.model} onClick={()=>void generate()}><Play size={15}/> {busy ? 'GENERATING' : 'GENERATE'}</button>
          </div>
        </section>

        <section className="panel generation-result-panel">
          <div className="panel-head">
            <div><div className="kicker">COMFYUI</div><div className="panel-title">IMAGE OUTPUT</div></div>
            <span className="provider-chip">{comfyStatus==='done' ? 'COMPLETE' : comfyStatus==='error' ? 'ERROR' : comfyStatus==='idle' ? 'IDLE' : comfyStatus.toUpperCase()}</span>
          </div>
          <div className="comfy-progress-wrap">
            <div className="comfy-progress-meta">
              <span>{comfyCurrentStep && comfyTotalSteps ? 'STEP ' + comfyCurrentStep + ' / ' + comfyTotalSteps : comfyStatus==='done' ? 'COMPLETE' : 'WAITING'}</span>
              <strong>{Math.round(comfyProgress)}%</strong>
            </div>
            <div className="comfy-progress-track"><div className="comfy-progress-fill" style={{width:comfyProgress + '%'}}/></div>
            {comfyCurrentNode && <div className="comfy-node">NODE {comfyCurrentNode}</div>}
          </div>

          {resultImage ? <div className="result-image-wrap">
            <img className="result-image" src={resultImage} alt={resultFilename || 'Generated result'}/>
            {resultFilename && <div className="result-filename">{resultFilename}</div>}
          </div> :
          <div className="result-placeholder"><WandSparkles size={22}/><span>{comfyStatus==='error' ? 'GENERATION FAILED' : 'NO IMAGE YET'}</span></div>}
        </section>
      </>}

      {tab==='history' && <section className="history-panel history-detail-panel">
        {!selectedHistory ? (
          <div className="empty-state"><History size={22}/><div><b>NO GENERATIONS</b><span>Completed generations will appear here.</span></div></div>
        ) : <>
          <div className="history-detail-head">
            <button className="ghost-btn" onClick={()=>setSelectedHistoryId('')}>ALL HISTORY</button>
            <div className="history-detail-meta">{new Date(selectedHistory.timestamp).toLocaleString()}</div>
          </div>

          <div className="history-detail-image-wrap">
            {selectedHistory.imageDataUrl
              ? <img className="history-detail-image" src={selectedHistory.imageDataUrl} alt={selectedHistory.imageFilename || 'Generated image'}/>
              : <div className="history-detail-no-image"><WandSparkles size={24}/><span>IMAGE NOT STORED</span></div>}
            {selectedHistory.imageFilename && <div className="result-filename">{selectedHistory.imageFilename}</div>}
          </div>

          <button className="secondary-btn history-settings-toggle" onClick={()=>setHistorySettingsVisible(v=>!v)}>
            <Settings2 size={13}/> {historySettingsVisible ? 'HIDE GENERATION SETTINGS' : 'VIEW GENERATION SETTINGS'}
          </button>

          {historySettingsVisible && selectedHistory.generationSettings && <section className="history-settings-card">
            <div className="history-section-title">GENERATION SETTINGS</div>
            <div className="history-grid">
              <div><span>PROVIDER</span><b>{selectedHistory.generationSettings.llm.provider}</b></div>
              <div><span>MODEL</span><b>{selectedHistory.generationSettings.llm.model || '—'}</b></div>
              <div><span>DEMOGRAPHIC</span><b>{selectedHistory.generationSettings.demographic.toUpperCase()}</b></div>
              <div><span>TEMPERATURE</span><b>{selectedHistory.generationSettings.llm.temperature.toFixed(2)}</b></div>
              <div><span>MAX TOKENS</span><b>{selectedHistory.generationSettings.llm.maxTokens}</b></div>
              <div><span>SIZE</span><b>{selectedHistory.generationSettings.width} × {selectedHistory.generationSettings.height}</b></div>
              <div><span>STEPS</span><b>{selectedHistory.generationSettings.steps}</b></div>
              <div><span>CFG</span><b>{selectedHistory.generationSettings.cfg}</b></div>
              <div><span>SAMPLER</span><b>{selectedHistory.generationSettings.sampler}</b></div>
              <div><span>MAX LORAS</span><b>{selectedHistory.generationSettings.maxLoras}</b></div>
            </div>

            <div className="history-section-title">SCENE</div>
            <div className="history-text-grid">
              {Object.entries(selectedHistory.generationSettings.constraints).map(([key,value])=><div key={key}><span>{key.toUpperCase()}</span><b>{String(value) || 'RANDOM'}</b></div>)}
            </div>

            <div className="history-section-title">SYSTEM PROMPT</div>
            <pre className="history-code">{selectedHistory.generationSettings.systemPrompt}</pre>
            <div className="history-section-title">DEMOGRAPHIC PROMPT</div>
            <pre className="history-code">{selectedHistory.generationSettings.demographicPrompts[selectedHistory.generationSettings.demographic]}</pre>
          </section>}

          <section className="history-settings-card">
            <div className="history-section-title">MODELS</div>
            <div className="history-model-header">
              {selectedHistory.checkpoint.thumbnail && thumbs[selectedHistory.checkpoint.id]
                ? <img src={thumbs[selectedHistory.checkpoint.id]} alt=""/>
                : <div className="history-model-placeholder"><Layers3 size={20}/></div>}
              <div><b>{selectedHistory.checkpoint.name}</b><span>{selectedHistory.checkpoint.baseModel || 'BASE UNKNOWN'}</span></div>
            </div>
            <div className="history-lora-detail">
              {selectedHistory.loras.map(l=><div className="history-lora-row" key={l.id}><span className={'stack-dot ' + (l.character ? 'character' : '')}/><div><b>{l.name}</b><span>{l.tags.join(', ') || 'NO TAGS'}</span></div><strong>{l.weight.toFixed(2)}</strong></div>)}
            </div>
          </section>

          <section className="history-settings-card">
            <div className="history-section-title">POSITIVE PROMPT</div>
            <div className="history-full-prompt">{selectedHistory.positivePrompt}</div>
            <div className="history-section-title">NEGATIVE PROMPT</div>
            <div className="history-full-prompt">{selectedHistory.negativePrompt}</div>
            {selectedHistory.rationale && <><div className="history-section-title">RATIONALE</div><div className="history-full-prompt">{selectedHistory.rationale}</div></>}
          </section>
        </>}

        {!selectedHistoryId && history.length>0 && <div className="history-list">
          {history.map(item=><button className="history-list-item" key={item.id} onClick={()=>openHistory(item.id)}>
            <div className="history-list-thumb">{item.imageDataUrl ? <img src={item.imageDataUrl} alt=""/> : <WandSparkles size={18}/>}</div>
            <div>
              <b>{new Date(item.timestamp).toLocaleString()}</b>
              <span>{item.checkpoint.name}</span>
              <small>{item.model || 'NO LLM MODEL'} · {item.loras.length} LORAS</small>
            </div>
            <span className="history-list-arrow">OPEN</span>
          </button>)}
        </div>}
      </section>}

      {error && <div className="error-box"><CircleAlert size={14}/><span>{error}</span></div>}
      {toast && <div className="toast">{toast}</div>}
      {webHostError && <div className="error-box"><CircleAlert size={14}/><span>{webHostError}</span></div>}
    </main>

    <aside className="right-rail">
      <div className="rail-label">MODELS</div>
      {selected && <div className="selected-model-card">
        <div className="selected-model-thumb">{thumbs[selected.id] ? <img src={thumbs[selected.id]} alt=""/> : <Layers3 size={22}/>}</div>
        <div className="selected-model-copy"><b>{selected.name}</b><span>{selected.baseModel || 'BASE UNKNOWN'}</span></div>
      </div>}

      <div className="right-section">
        <div className="right-section-head"><span>CHECKPOINTS</span><span>{filteredCheckpoints.length}</span></div>
        <div className="search-row compact"><Search size={13}/><input value={checkpointSearch} onChange={e=>setCheckpointSearch(e.target.value)} placeholder="Search checkpoints"/></div>
        <div className="checkpoint-list">
          {filteredCheckpoints.map(model=><button key={model.id} className={'checkpoint-row ' + (selected?.id===model.id ? 'selected' : '')} onClick={()=>selectCheckpoint(model.id)}>
            <div className="checkpoint-row-thumb">{thumbs[model.id] ? <img src={thumbs[model.id]} alt=""/> : <Layers3 size={16}/>}</div>
            <div className="checkpoint-row-copy"><b>{model.name}</b><span>{model.baseModel || 'BASE UNKNOWN'}</span><small>{model.tags.slice(0,3).join(' · ')}</small></div>
            {selected?.id===model.id && <Check size={14}/>}
          </button>)}
        </div>
      </div>

      <div className="right-section lora-section">
        <div className="right-section-head"><span>LORAS</span><span>{selectedLoraIds.length} / {compatibleLoras.length}</span></div>
        <div className="search-row compact"><Search size={13}/><input value={loraSearch} onChange={e=>setLoraSearch(e.target.value)} placeholder="Search LoRAs"/></div>
        <div className="lora-list">
          {filteredLoras.map(lora=><button key={lora.id} className={'lora-row ' + (selectedLoraIds.includes(lora.id) ? 'selected' : '')} onClick={()=>toggleLora(lora.id)}>
            <div className="lora-row-thumb">{thumbs[lora.id] ? <img src={thumbs[lora.id]} alt=""/> : <Layers3 size={15}/>}</div>
            <div className="lora-row-copy"><b>{lora.name}</b><span>{lora.character ? 'CHARACTER' : 'SUPPORT'}</span><small>{lora.tags.slice(0,3).join(' · ')}</small></div>
            {selectedLoraIds.includes(lora.id) && <Check size={14}/>}
          </button>)}
        </div>
      </div>

      <div className="right-section backend-section">
        <div className="right-section-head"><span>BACKEND</span></div>
        <label className="compact-field"><span>COMFYUI API</span><input value={comfyUrl} onChange={e=>setComfyUrl(e.target.value)}/></label>
        <label className="compact-field"><span>MODELS ROOT</span><input value={comfyRoot} onChange={e=>setComfyRoot(e.target.value)}/></label>
        <label className="compact-field"><span>RAPHAEL DB / CACHE</span><input value={raphaelRoot} onChange={e=>setRaphaelRoot(e.target.value)}/></label>
        <button className="secondary-btn full" onClick={()=>void scan()}><RefreshCw size={13}/> SCAN LIBRARY</button>
      </div>
    </aside>

          {settingsOpen && <div className="settings-overlay" onMouseDown={()=>setSettingsOpen(false)}>
        <div className="settings-drawer" onMouseDown={e=>e.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <div className="kicker">GENERATION</div>
            <div className="drawer-title">SETTINGS</div>
          </div>
          <button className="icon-btn" onClick={()=>setSettingsOpen(false)}><X size={16}/></button>
        </div>

        <div className="drawer-scroll">
          <section className="settings-section">
            <div className="settings-section-title">MODEL PROVIDER</div>
            <div className="provider-toggle">
              <button className={generationDraft.llm.provider==='ollama' ? 'active' : ''} onClick={()=>setGenerationDraft(d=>({...d,llm:{...d.llm,provider:'ollama',baseUrl:'http://127.0.0.1:11434'}}))}>OLLAMA</button>
              <button className={generationDraft.llm.provider==='openai-compatible' ? 'active' : ''} onClick={()=>setGenerationDraft(d=>({...d,llm:{...d.llm,provider:'openai-compatible',baseUrl:'http://127.0.0.1:8080/v1'}}))}>OPENAI COMPATIBLE</button>
            </div>
            <div className="field-grid">
              <label className="wide-field full-width"><span>BASE URL</span><input value={generationDraft.llm.baseUrl} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,baseUrl:e.target.value}}))}/></label>
              <label className="wide-field full-width"><span>API KEY</span><input type="password" value={generationDraft.llm.apiKey} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,apiKey:e.target.value}}))}/></label>
              <label className="wide-field full-width"><span>MODEL</span>
                <select value={generationDraft.llm.model} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,model:e.target.value}}))}>
                  <option value="">SELECT MODEL</option>{models.map(model=><option key={model}>{model}</option>)}
                </select>
              </label>
              <button className="secondary-btn full" onClick={()=>void (async()=>{
                try{
                  const found=await fetchModels(generationDraft.llm);
                  setGenerationDraft(d=>({...d,llm:{...d.llm,model:d.llm.model || found[0] || ''}}));
                }catch(e){setError(String(e));}
              })()}><RefreshCw size={13}/> GET MODELS</button>
              <label className="wide-field"><span>TEMPERATURE · {generationDraft.llm.temperature.toFixed(2)}</span><input type="range" min={0} max={2} step={0.05} value={generationDraft.llm.temperature} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,temperature:Number(e.target.value)}}))}/></label>
              <label className="wide-field"><span>MAX TOKENS</span><input type="number" min={128} max={16384} value={generationDraft.llm.maxTokens} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,maxTokens:Math.max(128,Number(e.target.value))}}))}/>
              <label className="wide-field"><span>CONTEXT TOKENS</span><input type="number" min={2048} max={131072} step={1024} value={generationDraft.llm.contextTokens} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,contextTokens:Math.max(2048,Number(e.target.value))}}))}/></label>
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
            <div className="settings-section-title">SCENE</div>
            <div className="field-grid">
              {(['setting','pose','expression','character','dress','composition','additional'] as const).map(key=>
                <label className={'wide-field ' + (key==='additional' ? 'full-width' : '')} key={key}>
                  <span>{key.replace('_',' ').toUpperCase()}</span>
                  {key==='additional'
                    ? <textarea className="settings-textarea" value={generationDraft.constraints[key]} onChange={e=>setGenerationDraft(d=>({...d,constraints:{...d.constraints,[key]:e.target.value}}))} placeholder="Additional prompt constraints"/>
                    : <input value={generationDraft.constraints[key]} onChange={e=>setGenerationDraft(d=>({...d,constraints:{...d.constraints,[key]:e.target.value}}))} placeholder={key==='character' ? 'Compatible character LoRA' : 'Random if blank'}/>}
                </label>
              )}
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">LORA LIMITS</div>
            <div className="field-grid">
              <label className="wide-field"><span>MAX LoRAs</span><input type="number" min={1} max={16} value={generationDraft.maxLoras} onChange={e=>setGenerationDraft(d=>({...d,maxLoras:Math.max(1,Number(e.target.value))}))}/></label>
              <label className="wide-field"><span>RANDOM MIN</span><input type="number" min={1} max={16} value={generationDraft.randomLoraMin} onChange={e=>setGenerationDraft(d=>({...d,randomLoraMin:Math.max(1,Number(e.target.value))}))}/></label>
              <label className="wide-field"><span>RANDOM MAX</span><input type="number" min={1} max={16} value={generationDraft.randomLoraMax} onChange={e=>setGenerationDraft(d=>({...d,randomLoraMax:Math.max(1,Number(e.target.value))}))}/></label>
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

        <div className="drawer-foot">
          <button className="secondary-btn" onClick={()=>setSettingsOpen(false)}>CANCEL</button>
          <button className="primary-btn" onClick={saveGenerationSettings}>SAVE SETTINGS</button>
        </div>
        </div>
      </div>}
  </div>
}

export default App;
