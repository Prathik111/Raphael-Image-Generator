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

const defaultDemographicPrompts: DemographicPrompts = {
  safe:'Keep all generated content non-sexual, non-explicit and suitable for general audiences. Avoid nudity and sexualized framing.',
  suggestive:'Allow mature, flirtatious or suggestive styling, but do not generate explicit sexual acts or graphic sexual detail.',
  explicit:'Allow adult sexual content between consenting adults, including nudity and explicit sexual detail, while avoiding minors or non-consensual scenarios.',
  'no-limits':'Do not add additional content restrictions beyond the application, model, platform and system requirements already in force.',
};


const demographicPromptsStorageKey = 'raphael-image-generator.demographic-prompts.v1';

function loadPersistedDemographicPrompts(): DemographicPrompts {
  if(typeof window === 'undefined') return defaultDemographicPrompts;
  try {
    const raw=window.localStorage.getItem(demographicPromptsStorageKey);
    if(!raw) return defaultDemographicPrompts;
    const parsed=JSON.parse(raw) as Partial<DemographicPrompts>;
    if(
      typeof parsed.safe !== 'string' ||
      typeof parsed.suggestive !== 'string' ||
      typeof parsed.explicit !== 'string' ||
      typeof parsed['no-limits'] !== 'string'
    ) return defaultDemographicPrompts;
    return {
      safe:parsed.safe,
      suggestive:parsed.suggestive,
      explicit:parsed.explicit,
      'no-limits':parsed['no-limits'],
    };
  } catch {
    return defaultDemographicPrompts;
  }
}

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

type ThumbnailState = 'loading' | 'ready' | 'error';

const thumbnailCache = new Map<string,string>();
const thumbnailPending = new Map<string,Promise<string>>();

function thumbnailReferences(reference?:string):string[]{
  if(!reference) return [];
  return [...new Set(
    reference
      .split('|')
      .map(value=>value.trim())
      .filter(Boolean),
  )];
}

async function fetchModelThumbnailReference(reference:string):Promise<string>{
  const cached = thumbnailCache.get(reference);
  if(cached) return cached;
  const pending = thumbnailPending.get(reference);
  if(pending) return pending;

  const request = (async()=>{
    let lastError:unknown;
    for(let attempt=0; attempt<3; attempt++){
      try{
        const url=await apiInvoke<string>('path_to_data_url',{path:reference});
        if(!url || !url.startsWith('data:image/')){
          throw new Error('Registry returned an invalid thumbnail response.');
        }
        thumbnailCache.set(reference,url);
        return url;
      }catch(error){
        lastError=error;
        if(attempt<2){
          await new Promise(resolve=>setTimeout(resolve,250 * (2 ** attempt)));
        }
      }
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError || 'Thumbnail request failed.'));
  })().finally(()=>thumbnailPending.delete(reference));

  thumbnailPending.set(reference,request);
  return request;
}

async function fetchModelThumbnail(references:string[]):Promise<string>{
  let lastError:unknown;
  for(const reference of references){
    try{
      return await fetchModelThumbnailReference(reference);
    }catch(error){
      lastError=error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error('No Registry thumbnail asset could be loaded.');
}

function ModelThumbnail({model,iconSize=18}:{model:LibrarySnapshot['checkpoints'][number];iconSize?:number}){
  const frameRef = useRef<HTMLSpanElement|null>(null);
  const references = useMemo(()=>thumbnailReferences(model.thumbnail),[model.thumbnail]);
  const referenceKey = references.join('|');
  const cachedReference = references.find(reference=>thumbnailCache.has(reference));
  const [state,setState] = useState<ThumbnailState>(
    cachedReference ? 'ready' : 'loading',
  );
  const [src,setSrc] = useState<string|null>(
    cachedReference ? thumbnailCache.get(cachedReference) || null : null,
  );

  useEffect(()=>{
    if(!references.length){
      setState('error');
      setSrc(null);
      return;
    }

    const cached=references.find(reference=>thumbnailCache.has(reference));
    if(cached){
      setSrc(thumbnailCache.get(cached) || null);
      setState('ready');
      return;
    }

    let active=true;
    let observer:IntersectionObserver|undefined;
    const load=()=>{
      void fetchModelThumbnail(references)
        .then(url=>{
          if(!active) return;
          setSrc(url);
          setState('ready');
        })
        .catch(()=>{
          if(!active) return;
          setSrc(null);
          setState('error');
        });
    };

    if(typeof IntersectionObserver==='undefined' || !frameRef.current){
      load();
    }else{
      observer=new IntersectionObserver(entries=>{
        if(!entries[0]?.isIntersecting) return;
        observer?.disconnect();
        load();
      },{rootMargin:'160px'});
      observer.observe(frameRef.current);
    }

    return ()=>{
      active=false;
      observer?.disconnect();
    };
  },[referenceKey]);

  const handleImageError=()=>{
    const failedReference=references.find(reference=>thumbnailCache.get(reference)===src);
    if(failedReference) thumbnailCache.delete(failedReference);
    setState('loading');
    void fetchModelThumbnail(references)
      .then(url=>{
        setSrc(url);
        setState('ready');
      })
      .catch(()=>{
        setSrc(null);
        setState('error');
      });
  };

  return <span ref={frameRef} className={'model-thumbnail ' + state}
    aria-label={state==='error' ? 'Thumbnail unavailable' : state==='ready' ? model.name : 'Loading thumbnail'}
    title={state==='error' ? 'Thumbnail unavailable' : model.name}>
    {src && state==='ready'
      ? <img src={src} alt="" loading="lazy" decoding="async" onError={handleImageError}/>
      : <Layers3 size={iconSize} aria-hidden="true"/>}
  </span>;
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
  // Automatic/random LoRA selection is based on the checkpoint's exact
  // base-model tag. Do not use checkpoint names or fuzzy substring matches:
  // those can make an Illustrus LoRA look compatible with an Anima checkpoint.
  const base=normUi(checkpoint.baseModel || '');
  if(base.length>2){
    return lora.tags.some(tag=>normUi(tag)===base);
  }

  // Fallback for older Registry entries without baseModel metadata.
  const checkpointTags=new Set(
    checkpoint.tags.map(normUi).filter(tag=>tag.length>2),
  );
  return lora.tags.some(tag=>checkpointTags.has(normUi(tag)));
}

function isCharacterLoraForCheckpoint(
  lora:LibrarySnapshot['loras'][number],
  checkpoint:LibrarySnapshot['checkpoints'][number],
){
  const characterTagged = lora.character || lora.tags.some(tag=>normUi(tag)==='character');
  if(!characterTagged) return false;

  const baseKeys=[checkpoint.baseModel || '',...checkpoint.tags]
    .map(normUi)
    .filter(x=>x.length>2);

  if(!baseKeys.length) return true;

  const loraKeys=[lora.baseModel || '',...lora.tags]
    .map(normUi)
    .filter(x=>x.length>2);

  return loraKeys.some(loraKey=>
    baseKeys.some(baseKey=>
      loraKey===baseKey || loraKey.includes(baseKey) || baseKey.includes(loraKey)
    )
  );
}

function promptNeedsExpansion(pair:PromptPair){
  const positive=pair.positive_prompt.trim();
  const words=positive.split(/\s+/).filter(Boolean).length;
  const clauses=positive.split(',').map(x=>x.trim()).filter(Boolean).length;
  const sceneSignals=[
    /pose|posture|standing|sitting|lying|walking|kneeling/i,
    /expression|smile|frown|serious|calm|happy|sad|angry|confident|gaze|looking/i,
    /background|environment|scene|room|street|forest|sky|wall|landscape|interior|exterior/i,
    /lighting|light|shadow|rim light|sunlight|moonlight|neon/i,
    /camera|close-up|medium shot|wide shot|portrait|three-quarter|full body|perspective/i,
  ].filter(pattern=>pattern.test(positive)).length;
  return words < 90 || clauses < 18 || sceneSignals < 4;
}

function App(){
  const [provider,setProvider]=useState<ProviderKind>('ollama');
  const [llm,setLlm]=useState<LlmSettings>({
    provider:'ollama',
    baseUrl:'http://127.0.0.1:11434',
    apiKey:'',
    model:'',
    temperature:0.72,
    maxTokens:8192,
    contextTokens:32768,
  });
  const [models,setModels]=useState<string[]>([]);
  const [library,setLibrary]=useState<LibrarySnapshot|null>(null);
  const [selectedId,setSelectedId]=useState('');
  const [comfyRoot,setComfyRoot]=useState('D:\\ComfyUI\\models');
  const [registryUrl,setRegistryUrl]=useState('');
  const [comfyUrl,setComfyUrl]=useState('http://127.0.0.1:8188');

  const [constraints,setConstraints]=useState<Constraints>(emptyConstraints);
  const [demographic,setDemographic]=useState<DemographicLevel>('safe');
  const [demographicPrompts,setDemographicPrompts]=useState<DemographicPrompts>(loadPersistedDemographicPrompts);
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
  const [selectedLoraIds,setSelectedLoraIds]=useState<string[]>([]);
  const [manualLoraIds,setManualLoraIds]=useState<string[]>([]);

  const [settingsOpen,setSettingsOpen]=useState(false);
  const [generationDraft,setGenerationDraft]=useState({
    llm,
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

  const allLoras=useMemo(
    ()=>library?.loras || [],
    [library],
  );

  // Compatibility is used for automatic/random selection only.
  // The selector itself exposes the complete Registry LoRA inventory.
  const compatibleLoras=useMemo(
    ()=>library && selected ? allLoras.filter(lora=>isCompatibleLoraUi(lora,selected)) : [],
    [allLoras,library,selected],
  );

  const manualLoras=useMemo(
    ()=>allLoras.filter(lora=>manualLoraIds.includes(lora.id)),
    [allLoras,manualLoraIds],
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
    return allLoras.filter(lora =>
      !q || [lora.name,lora.baseModel || '',...lora.tags].join(' ').toLowerCase().includes(q)
    );
  },[allLoras,loraSearch]);

  useEffect(()=>{
    void loadHistory();
    void discoverRoots();
    if(isTauriRuntime){
      void refreshModels();
    }
  },[]);


  const setStageStatus=(key:Stage,value:Status)=>{
    setStatus(x=>({...x,[key]:value}));
  };

  async function discoverRoots(){
    try{
      const config=await apiInvoke<{models_root:string|null;registry_url:string|null}>('discover_raphael_config');
      if(config.models_root) setComfyRoot(config.models_root);
      if(config.registry_url) setRegistryUrl(config.registry_url);

      if(!isTauriRuntime){
        const hostLlm=await apiInvoke<{provider:ProviderKind;baseUrl:string;model:string}>('get_host_llm_config');
        const nextLlm={...llm,provider:hostLlm.provider,baseUrl:hostLlm.baseUrl,model:hostLlm.model || ''};
        setLlm(nextLlm);
        setGenerationDraft(d=>({...d,llm:{...d.llm,provider:hostLlm.provider,baseUrl:hostLlm.baseUrl,model:hostLlm.model || ''}}));
        await fetchModels(nextLlm);
      }

      await scan(config.models_root || comfyRoot,config.registry_url || undefined);
    }catch(e){
      setError(String(e));
    }
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

  async function scan(rootOverride=comfyRoot,registryOverride=registryUrl || undefined){
    setError('');
    setStage('library');
    setStageStatus('library','running');
    try{
      const snap=await apiInvoke<LibrarySnapshot>('scan_library',{
        req:{comfyRoot:rootOverride,registryUrl:registryOverride || null},
      });
      setLibrary(snap);
      if(!selectedId && snap.checkpoints[0]) setSelectedId(snap.checkpoints[0].id);
      const ids=new Set(snap.loras.map(x=>x.id));
      setSelectedLoraIds(current=>current.filter(id=>ids.has(id)));
      setManualLoraIds(current=>current.filter(id=>ids.has(id)));
      setStageStatus('library','done');
      if(snap.warnings.length){
        setError(snap.warnings.join(' '));
      }
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
    setDemographic(draft.demographic);
    const savedDemographicPrompts={...draft.demographicPrompts};
    setDemographicPrompts(savedDemographicPrompts);
    try{
      window.localStorage.setItem(demographicPromptsStorageKey,JSON.stringify(savedDemographicPrompts));
    }catch{}
    setMaxLoras(cappedMax);
    setWidth(Math.max(64,draft.width));
    setHeight(Math.max(64,draft.height));
    setSteps(Math.max(1,draft.steps));
    setCfg(Math.max(0,draft.cfg));
    setSampler(draft.sampler || 'euler');
    if(webHost && isTauriRuntime){
      void apiInvoke('update_web_host_llm',{llmSettings:draft.llm}).catch(e=>setWebHostError(String(e)));
    }
    setSettingsOpen(false);
    setToast('Generation settings saved');
  }

  async function rollStack(){
    if(!selected || !library){
      setError('Select a checkpoint first.');
      return null;
    }
    setError('');

    const manualIds=manualLoraIds.filter(id=>allLoras.some(lora=>lora.id===id));
    const manualHasCharacter=manualIds.some(id=>{
      const lora=allLoras.find(item=>item.id===id);
      return !!lora && isCharacterLoraForCheckpoint(lora,selected);
    });

    const minCount=Math.max(1,Math.min(maxLoras,constraints.randomLoraMin));
    const maxCount=Math.max(minCount,Math.min(maxLoras,constraints.randomLoraMax));
    const randomTarget=Math.floor(Math.random()*(maxCount-minCount+1))+minCount;

    const requiredCount=Math.max(
      randomTarget,
      manualIds.length + (manualHasCharacter ? 0 : 1),
    );
    if(requiredCount>maxLoras){
      setError('Increase MAX LoRAs by at least one slot so the random stack can include a character LoRA for the selected base model.');
      return null;
    }

    // compatibleLoras only contains LoRAs carrying the checkpoint's exact
    // base-model tag, e.g. "anima" for an Anima checkpoint.
    const failedRandomIds=new Set<string>();
    const randomCandidateCount=compatibleLoras.filter(
      lora=>!manualIds.includes(lora.id),
    ).length;
    const maxAttempts=Math.min(8,Math.max(1,randomCandidateCount));
    let lastError:unknown=null;

    setPrepared(null);
    setPrompts(null);
    setStage('compatibility');
    setStageStatus('compatibility','running');

    for(let attempt=0; attempt<maxAttempts; attempt++){
      const characterPool=compatibleLoras
        .filter(lora=>!manualIds.includes(lora.id))
        .filter(lora=>!failedRandomIds.has(lora.id))
        .filter(lora=>isCharacterLoraForCheckpoint(lora,selected))
        .sort(()=>Math.random()-0.5);

      const availablePool=compatibleLoras
        .filter(lora=>!manualIds.includes(lora.id))
        .filter(lora=>!failedRandomIds.has(lora.id))
        .filter(lora=>!isCharacterLoraForCheckpoint(lora,selected))
        .sort(()=>Math.random()-0.5);

      const randomSlots=Math.max(
        0,
        Math.min(maxLoras,requiredCount)-manualIds.length,
      );
      const randomIds:string[]=[];

      if(!manualHasCharacter){
        const character=characterPool[0];
        if(!character){
          lastError=new Error('No compatible character LoRA was found for the selected base-model tag in the Registry.');
          break;
        }
        randomIds.push(character.id);
      }

      const remainingSlots=Math.max(0,randomSlots-randomIds.length);
      randomIds.push(...availablePool.slice(0,remainingSlots).map(lora=>lora.id));

      const combinedIds=[...manualIds,...randomIds];

      try{
        const result=await apiInvoke<PreparedGeneration>('prepare_generation',{
          req:{
            checkpoint:selected,
            loras:library.loras,
            selectedLoraIds:combinedIds,
            ...constraints,
            randomLoraMax:Math.min(maxLoras,constraints.randomLoraMax),
            registryUrl:registryUrl || null,
          },
        });
        setSelectedLoraIds(combinedIds);
        setPrepared(result);
        setStageStatus('compatibility','done');
        setStage('selection');
        setStageStatus('selection','done');
        return result;
      }catch(e){
        lastError=e;
        // Never silently replace manually selected LoRAs. Only discard
        // automatically randomized candidates and try another match.
        for(const id of randomIds){
          failedRandomIds.add(id);
        }
        if(randomIds.length===0) break;
      }
    }

    setStageStatus('compatibility','error');
    setStageStatus('selection','error');
    setError(String(lastError || 'No compatible randomized LoRA stack could be prepared for the selected checkpoint.'));
    return null;
  }

  function toggleLora(id:string){
    const alreadySelected=selectedLoraIds.includes(id);
    if(alreadySelected){
      setSelectedLoraIds(current=>current.filter(x=>x!==id));
      setManualLoraIds(current=>current.filter(x=>x!==id));
    }else{
      setSelectedLoraIds(current=>[...current,id]);
      setManualLoraIds(current=>[...current,id]);
    }
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
          registryUrl:registryUrl || null,
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
        'DESCRIPTION: ' + (l.description || '(none)') + '\n' +
        'ACTIVATION PROMPT(S): ' + (l.activationTags.length ? l.activationTags.join(' | ') : '(none)')
      ).join('\n\n');

      // The selected demographic prompt is the complete system prompt.
      // Nothing else is prepended, appended, or merged into it.
      const selectedSystemPrompt = demographicPrompts[demographic];

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
        'Write the FINAL positive and negative prompts that will be sent directly to the image model. The positive prompt must be one long, coherent, comma-separated tag string of 90-160 words with at least 18 meaningful visual tags or short phrases. Use the selected LoRA metadata as actual prompt-building input, not as reference-only information. For every documented activation prompt, include the exact activation phrase in the positive prompt at least once, unchanged, and place it naturally next to the visual concept it activates instead of collecting activation prompts at the end. The LoRA description explains what visual concept the activation prompt controls; use that description to decide where and how that activation phrase belongs. Do not invent or paraphrase activation prompts, and do not omit them. Do not expose LoRA implementation syntax such as <lora:...> or weighted [LoRA - ...] notation. Do not mention checkpoint/model names, filenames or base-model labels. Explicitly cover subject state/action, pose, hands/arms, head direction, gaze, facial expression, emotional state, clothing/accessories, interaction, setting, background/environment, atmosphere, camera viewpoint, framing, perspective, depth, lighting, color/mood, materials and finishing details using short literal tags or compact phrases only. Never turn these into sentences or metaphors. The negative prompt should be a useful 20-35 item comma-separated list of short concrete failure tags targeted to the actual image and selected LoRAs. Return JSON only.';

      await streamLlm({
        settings:llm,
        systemPrompt:selectedSystemPrompt,
        userPrompt,
      },event=>{
        streamText.current+=event;
        flushSync(()=>setStream(streamText.current));
      });

      let rawPair=await apiInvoke<PromptPair>('parse_prompt_pair',{raw:streamText.current});

      if(promptNeedsExpansion(rawPair)){
        streamText.current='';
        flushSync(()=>setStream(''));
        const expansionPrompt=
          'EXPANSION PASS. Rewrite the previous result as the final production prompt pair in deterministic tag format. Preserve every required scene constraint and every documented LoRA activation prompt. Each activation prompt must appear exactly as documented, at least once, beside the visual concept described by its LoRA, not as an appended block at the end. Use the LoRA descriptions to understand the intended visual effect. Do not remove or paraphrase activation prompts. Remove checkpoint names, model filenames, LoRA names and implementation syntax such as <lora:...> or weighted [LoRA - ...] notation. Rewrite the positive prompt as ONE long, coherent comma-separated tag string of 90-160 words with at least 18 meaningful visual tags or short phrases. No sentences, metaphors, storytelling, poetic language or sentence punctuation. Use literal canonical tags, mostly 1-5 words each. Explicitly include subject identity, visible appearance, current state, action/activity, body pose, hands/arms, head direction, gaze, facial expression, emotional state, clothing/accessories, interaction with surroundings, setting, background/environment, atmosphere, camera viewpoint, shot type, framing, perspective, depth, lighting direction/quality, color/mood, materials/textures and finishing details. Do not rely on any LoRA to provide expression, pose, state, background or composition. Also provide a targeted 20-35 item negative tag list. Return JSON only.';
        await streamLlm({
          settings:llm,
          systemPrompt:selectedSystemPrompt,
          userPrompt:
            expansionPrompt +
            '\n\nSELECTED LoRA METADATA:\n' + loraMetadata +
            '\n\nPREVIOUS JSON:\n' + JSON.stringify(rawPair),
        },event=>{
          streamText.current+=event;
          flushSync(()=>setStream(streamText.current));
        });
        rawPair=await apiInvoke<PromptPair>('parse_prompt_pair',{raw:streamText.current});
      }

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
        // Keep the history record explicit about the exact system prompt that was used.
        systemPrompt:demographicPrompts[demographic],
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
      const info=await apiInvoke<WebHostInfo>('start_web_host',{port:1424,llmSettings:llm});
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

      {tab==='generate' && <div className="generate-layout">
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
        </div>}

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
              <div><span>CONTEXT TOKENS</span><b>{selectedHistory.generationSettings.llm.contextTokens || 16384}</b></div>
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
              <ModelThumbnail model={selectedHistory.checkpoint} iconSize={20}/>
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
        <div className="selected-model-thumb"><ModelThumbnail model={selected} iconSize={22}/></div>
        <div className="selected-model-copy"><b>{selected.name}</b><span>{selected.baseModel || 'BASE UNKNOWN'}</span></div>
      </div>}

      <div className="right-section">
        <div className="right-section-head"><span>CHECKPOINTS</span><span>{filteredCheckpoints.length}</span></div>
        <div className="search-row compact"><Search size={13}/><input value={checkpointSearch} onChange={e=>setCheckpointSearch(e.target.value)} placeholder="Search checkpoints"/></div>
        <div className="checkpoint-list">
          {filteredCheckpoints.map(model=><button key={model.id} title={[model.name,model.baseModel || '',...model.tags].filter(Boolean).join(' · ')} className={'checkpoint-row ' + (selected?.id===model.id ? 'selected' : '')} onClick={()=>selectCheckpoint(model.id)}>
            <div className="checkpoint-row-thumb"><ModelThumbnail model={model} iconSize={16}/></div>
            <div className="checkpoint-row-copy"><b>{model.name}</b><span>{model.baseModel || 'BASE UNKNOWN'}</span><small>{model.tags.slice(0,3).join(' · ')}</small></div>
            {selected?.id===model.id && <Check size={14}/>}
          </button>)}
        </div>
      </div>

      <div className="right-section lora-section">
        <div className="right-section-head"><span>LORAS</span><span>{selectedLoraIds.length} / {allLoras.length}</span></div>
        <div className="search-row compact"><Search size={13}/><input value={loraSearch} onChange={e=>setLoraSearch(e.target.value)} placeholder="Search LoRAs"/></div>
        <div className="lora-list">
          {filteredLoras.map(lora=><button key={lora.id} title={[lora.name,lora.baseModel || '',...lora.tags].filter(Boolean).join(' · ')} className={'lora-row ' + (selectedLoraIds.includes(lora.id) ? 'selected' : '')} onClick={()=>toggleLora(lora.id)}>
            <div className="lora-row-thumb"><ModelThumbnail model={lora} iconSize={15}/></div>
            <div className="lora-row-copy"><b>{lora.name}</b><span>{lora.character ? 'CHARACTER' : 'SUPPORT'}</span><small>{lora.tags.slice(0,3).join(' · ')}</small></div>
            {selectedLoraIds.includes(lora.id) && <Check size={14}/>}
          </button>)}
        </div>
      </div>

      <details className="right-section backend-section">
        <summary><span>BACKEND</span><span>CONFIGURE</span></summary>
        <label className="compact-field"><span>COMFYUI API</span><input value={comfyUrl} onChange={e=>setComfyUrl(e.target.value)}/></label>
        <label className="compact-field"><span>MODELS ROOT</span><input value={comfyRoot} onChange={e=>setComfyRoot(e.target.value)}/></label>
        <label className="compact-field"><span>REGISTRY URL</span><input value={registryUrl} onChange={e=>setRegistryUrl(e.target.value)}/></label>
        <button className="secondary-btn full" onClick={()=>void scan()}><RefreshCw size={13}/> SCAN LIBRARY</button>
      </details>
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
              <label className="wide-field"><span>MAX TOKENS</span><input type="number" min={128} max={16384} value={generationDraft.llm.maxTokens} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,maxTokens:Math.max(128,Number(e.target.value))}}))}/></label>
              <label className="wide-field"><span>CONTEXT TOKENS</span><input type="number" min={2048} max={131072} step={1024} value={generationDraft.llm.contextTokens} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,contextTokens:Math.max(2048,Number(e.target.value))}}))}/></label>
            </div>
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
                  <span>{level.replace('-',' ').toUpperCase()} SYSTEM PROMPT · SENT EXACTLY TO LLM</span>
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
