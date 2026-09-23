import { useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { Channel, invoke } from '@tauri-apps/api/core';
import { Check, CircleAlert, Copy, Database, FolderOpen, History, Layers3, Play, RefreshCw, Settings2, Sparkles, Terminal, WandSparkles, X } from 'lucide-react';
import type { Constraints, GenerationRecord, LibrarySnapshot, LlmSettings, PreparedGeneration, PromptPair, ProviderKind } from './types';

type Stage = 'library' | 'compatibility' | 'selection' | 'llm' | 'workflow' | 'comfy' | 'recorded';
type Status = 'idle' | 'running' | 'done' | 'error';

function normUi(value: string) {
  return value.trim().toLowerCase().replace(/[ _\-./]+/g, '');
}

function isCompatibleLoraUi(lora: LibrarySnapshot['loras'][number], checkpoint: LibrarySnapshot['checkpoints'][number]) {
  const keys = [...checkpoint.tags, checkpoint.baseModel || '', checkpoint.name]
    .map(normUi)
    .filter(x => x.length > 2);
  const explicit = !!lora.baseModel && keys.some(k => {
    const b = normUi(lora.baseModel || '');
    return k === b || k.includes(b) || b.includes(k);
  });
  const tagged = lora.tags.some(tag => {
    const t = normUi(tag);
    return keys.some(k => k === t || k.includes(t) || t.includes(k));
  });
  return explicit || tagged;
}

const stages: Array<{key: Stage; label: string}> = [
  {key:'library',label:'LIBRARY'},{key:'compatibility',label:'COMPATIBILITY'},
  {key:'selection',label:'RANDOM STACK'},{key:'llm',label:'LLM PROMPT'},
  {key:'workflow',label:'WORKFLOW TOOL'},{key:'comfy',label:'COMFYUI'},{key:'recorded',label:'RECORDED'},
];

const emptyConstraints: Constraints = {
  setting:'', pose:'', expression:'', character:'', dress:'', composition:'', additional:'',
  randomLoraMin:2, randomLoraMax:4,
};

function App() {
  const [provider, setProvider] = useState<ProviderKind>('ollama');
  const [llm, setLlm] = useState<LlmSettings>({
    provider:'ollama', baseUrl:'http://127.0.0.1:11434', apiKey:'', model:'', temperature:0.75, maxTokens:1200,
  });
  const [models, setModels] = useState<string[]>([]);
  const [library, setLibrary] = useState<LibrarySnapshot | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [comfyRoot, setComfyRoot] = useState('D:\\ComfyUI\\models');
  const [raphaelRoot, setRaphaelRoot] = useState('');
  const [comfyUrl, setComfyUrl] = useState('http://127.0.0.1:8188');
  const [constraints, setConstraints] = useState<Constraints>(emptyConstraints);
  const [prepared, setPrepared] = useState<PreparedGeneration | null>(null);
  const [prompts, setPrompts] = useState<PromptPair | null>(null);
  const [stream, setStream] = useState('');
  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [tab, setTab] = useState<'generate'|'history'|'settings'>('generate');
  const [stage, setStage] = useState<Stage>('library');
  const [status, setStatus] = useState<Record<Stage, Status>>({
    library:'idle', compatibility:'idle', selection:'idle', llm:'idle',
    workflow:'idle', comfy:'idle', recorded:'idle',
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [toast, setToast] = useState('');
  const [thumbs, setThumbs] = useState<Record<string,string>>({});
  const [selectedLoraIds, setSelectedLoraIds] = useState<string[]>([]);
  const streamText = useRef('');

  const selected = useMemo(
    () => library?.checkpoints.find(x => x.id === selectedId) || library?.checkpoints[0],
    [library, selectedId],
  );

  const compatibleLoras = useMemo(
    () => (library && selected ? library.loras.filter(lora => isCompatibleLoraUi(lora, selected)) : []),
    [library, selected],
  );

  const manualLoras = useMemo(
    () => compatibleLoras.filter(lora => selectedLoraIds.includes(lora.id)),
    [compatibleLoras, selectedLoraIds],
  );

  useEffect(() => {
    void loadHistory();
    void refreshModels();
    void discoverRoots();
    return () => undefined;
  }, []);

  useEffect(() => { setLlm(x => ({...x, provider})); }, [provider]);

  useEffect(() => {
    if (!library) return;
    for (const model of [...library.checkpoints, ...library.loras].slice(0, 80)) {
      if (!model.thumbnail || thumbs[model.id]) continue;
      void invoke<string>('path_to_data_url', {path:model.thumbnail})
        .then(url => setThumbs(x => ({...x, [model.id]:url})))
        .catch(() => undefined);
    }
  }, [library]);

  const setStageStatus = (key: Stage, value: Status) => {
    setStatus(x => ({...x, [key]:value}));
  };

  async function discoverRoots() {
    try {
      const config = await invoke<{models_root: string | null; db_path: string | null}>('discover_raphael_config');
      if (config.models_root) setComfyRoot(config.models_root);
      if (config.db_path) setRaphaelRoot(config.db_path);
      if (config.models_root) {
        await scan(config.models_root, config.db_path || undefined);
      }
    } catch {}
  }

  async function loadHistory() {
    try {
      const records = await invoke<Array<{payload: GenerationRecord}>>('load_history');
      setHistory(records.map(x => x.payload));
    } catch {}
  }

  async function refreshModels() {
    try {
      const found = await invoke<string[]>('list_provider_models', {settings:llm});
      setModels(found);
      if (!llm.model && found[0]) setLlm(x => ({...x, model:found[0]}));
    } catch (e) {
      setError(String(e));
    }
  }

  async function scan(rootOverride = comfyRoot, raphaelOverride = raphaelRoot || undefined) {
    setError('');
    setStage('library');
    setStageStatus('library','running');
    try {
      const snap = await invoke<LibrarySnapshot>('scan_library', {
        req:{comfyRoot:rootOverride, raphaelRoot:raphaelOverride || null},
      });
      setLibrary(snap);
      if (snap.checkpoints[0]) setSelectedId(snap.checkpoints[0].id);
      setStageStatus('library','done');
      setToast(snap.checkpoints.length + ' checkpoints · ' + snap.loras.length + ' LoRAs');
    } catch (e) {
      setStageStatus('library','error');
      setError(String(e));
    }
  }

  function updateConstraint<K extends keyof Constraints>(key:K, value:Constraints[K]) {
    setConstraints(x => ({...x, [key]:value}));
  }

  async function rollStack() {
    if (!selected || !library) {
      setError('Scan the ComfyUI library and select a checkpoint first.');
      return null;
    }
    setError('');
    setStage('compatibility');
    setStageStatus('compatibility','running');
    try {
      const result = await invoke<PreparedGeneration>('prepare_generation', {
        req:{checkpoint:selected, loras:library.loras, selectedLoraIds:[], ...constraints},
      });
      setPrepared(result);
      setSelectedLoraIds(result.loras.map(lora => lora.id));
      setStageStatus('compatibility','done');
      setStage('selection');
      setStageStatus('selection','done');
      return result;
    } catch (e) {
      setStageStatus('compatibility','error');
      setStageStatus('selection','error');
      setError(String(e));
      return null;
    }
  }

  function toggleLora(id:string) {
    setSelectedLoraIds(current => current.includes(id)
      ? current.filter(x => x !== id)
      : [...current, id]);
    setPrepared(null);
    setPrompts(null);
  }

  async function generate() {
    if (busy || !selected || !library) return;
    setBusy(true);
    setError('');
    setToast('');
    try {
      setStage('compatibility');
      setStageStatus('compatibility','running');
      const prep = await invoke<PreparedGeneration>('prepare_generation', {
        req:{checkpoint:selected, loras:library.loras, selectedLoraIds, ...constraints},
      });
      setPrepared(prep);
      setStageStatus('compatibility','done');
      setStage('selection');
      setStageStatus('selection','done');

      setStage('llm');
      setStageStatus('llm','running');
      streamText.current = '';
      setStream('');
      setPrompts(null);

      const onEvent = new Channel<{text:string}>();
      onEvent.onmessage = event => {
        streamText.current += event.text;
        flushSync(() => {
          setStream(streamText.current);
        });
      };

      const loraMetadata = prep.loras.map((l, index) =>
        'LORA ' + (index + 1) + '\n' +
        'NAME: ' + l.name + '\n' +
        'TYPE: ' + (l.character ? 'CHARACTER IDENTITY' : 'SUPPORTING CONCEPT') + '\n' +
        'BASE MODEL: ' + (l.baseModel || 'unknown') + '\n' +
        'TAGS: ' + (l.tags.length ? l.tags.join(', ') : '(none)') + '\n' +
        'DESCRIPTION: ' + (l.description || '(none)')
      ).join('\n\n');

      const systemPrompt =
        'You are a Stable Diffusion prompt planner. ' +
        'Understand the purpose of every selected LoRA from its name, type, tags and description before writing the prompt. ' +
        'Use each LoRA only for concepts it plausibly provides. ' +
        'The character LoRA is the sole authority for character identity, appearance and named-character traits. ' +
        'Supporting LoRAs can contribute only their documented visual concept/style/content. ' +
        'Do not invent character identities or unsupported LoRA effects. ' +
        'Do NOT output LoRA activation/trigger words, angle-bracket LoRA syntax, weights, or implementation details in the positive prompt. ' +
        'The application will deterministically append activation triggers after you finish. ' +
        'Return JSON only with positive_prompt, negative_prompt, rationale.';

      const userPrompt =
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

      await invoke('stream_llm', {
        req:{
          settings:llm, systemPrompt, userPrompt,
        },
        onEvent,
      });

      const rawPair = await invoke<PromptPair>('parse_prompt_pair', {raw:streamText.current});
      const pair = await invoke<PromptPair>('finalize_prompt_pair', {
        req:{promptPair:rawPair, loras:prep.loras},
      });
      setPrompts(pair);
      setStageStatus('llm','done');

      setStage('workflow');
      setStageStatus('workflow','running');
      const workflow = await invoke<Record<string,unknown>>('build_workflow', {
        req:{
          checkpoint:prep.checkpoint, loras:prep.loras, width:1024, height:1024,
          steps:28, cfg:6.5, sampler:'euler', seed:Math.floor(Math.random() * 2147000000),
        },
      });
      const injected = await invoke<Record<string,unknown>>('inject_prompts', {
        req:{
          workflow,
          positivePrompt:pair.positive_prompt,
          negativePrompt:pair.negative_prompt,
        },
      });
      setStageStatus('workflow','done');

      setStage('comfy');
      setStageStatus('comfy','running');
      let promptId: string | undefined;
      try {
        const response = await invoke<{prompt_id?:string}>('submit_to_comfy', {
          req:{comfyUrl, workflow:injected},
        });
        promptId = response.prompt_id;
        setStageStatus('comfy','done');
      } catch (e) {
        setStageStatus('comfy','error');
        setError('ComfyUI submission failed: ' + String(e));
      }

      const record: GenerationRecord = {
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
      await invoke('append_history', {payload:record});
      setHistory(x => [record, ...x].slice(0,100));
      setStage('recorded');
      setStageStatus('recorded','done');
      setToast(promptId ? 'Submitted to ComfyUI · ' + promptId : 'Generation recorded');
    } catch (e) {
      setError(String(e));
      setStageStatus(stage,'error');
    } finally {
      setBusy(false);
    }
  }

  async function pickFolder(setter:(v:string)=>void) {
    const picked = await invoke<{path?:string|null}>('pick_folder');
    if (picked.path) setter(picked.path);
  }

  async function copy(text:string) {
    await navigator.clipboard?.writeText(text);
    setToast('Copied');
  }

  return <div className="app-shell">
    <iframe className="raphael-bg" src="/raphael-background.html" title="Raphael background" aria-hidden="true"/>
    <div className="vignette"/>

    <header className="topbar">
      <div className="brand">
        <div className="brand-mark"><Sparkles size={15}/></div>
        <div><div className="brand-title">PROMPT FORGE</div><div className="brand-sub">RAPHAEL GENERATION ENGINE</div></div>
      </div>
      <div className="top-meta"><span className="pulse-dot"/><span>{busy ? 'STREAMING' : 'READY'}</span><span>{library?.checkpoints.length || 0} CHECKPOINTS</span><span>{library?.loras.length || 0} LoRAs</span></div>
    </header>

    <aside className="left-rail">
      <div className="rail-label">WORKSPACE</div>
      <button className={'rail-btn ' + (tab === 'generate' ? 'active' : '')} onClick={() => setTab('generate')}><WandSparkles size={15}/> GENERATE</button>
      <button className={'rail-btn ' + (tab === 'history' ? 'active' : '')} onClick={() => setTab('history')}><History size={15}/> HISTORY <span>{history.length}</span></button>
      <button className={'rail-btn ' + (tab === 'settings' ? 'active' : '')} onClick={() => setTab('settings')}><Settings2 size={15}/> SETTINGS</button>
      <div className="rail-spacer"/>
      <div className="root-box"><div className="root-label">COMFYUI ROOT</div><div className="root-path">{comfyRoot}</div><button onClick={() => void pickFolder(setComfyRoot)}><FolderOpen size={13}/> BROWSE</button></div>
    </aside>

    <main className="content">
      <section className="stage-strip">
        {stages.map((s, i) => <div className={'stage ' + status[s.key] + ' ' + (stage === s.key ? 'current' : '')} key={s.key}>
          <div className="stage-index">{status[s.key] === 'done' ? <Check size={12}/> : status[s.key] === 'error' ? <CircleAlert size={12}/> : i + 1}</div>
          <div className="stage-label">{s.label}</div>
        </div>)}
      </section>

      {tab === 'generate' && <>
        <section className="hero-section">
          <div className="section-head">
            <div><div className="kicker">CHECKPOINT LIBRARY / RAPHAEL CACHE</div><div className="section-title">SELECT BASE MODEL</div></div>
            <div className="head-actions"><button onClick={() => void scan()}><RefreshCw size={13}/> SCAN</button></div>
          </div>
          {!library?.checkpoints.length
            ? <div className="empty-state"><Database size={22}/><div><b>NO CHECKPOINT LIBRARY LOADED</b><span>Set your ComfyUI models folder and scan it.</span></div></div>
            : <div className="checkpoint-grid">{library.checkpoints.map(model =>
              <button key={model.id} className={'model-card ' + (selected?.id === model.id ? 'selected' : '')} onClick={() => {setSelectedId(model.id); setSelectedLoraIds([]); setPrepared(null); setPrompts(null);}}>
                <div className="model-thumb">
                  {thumbs[model.id] ? <img src={thumbs[model.id]} alt=""/> : <div className="thumb-fallback"><Layers3 size={28}/><span>{(model.baseModel || model.name).slice(0,18).toUpperCase()}</span></div>}
                  <div className="thumb-overlay"><span>{model.baseModel || 'BASE UNKNOWN'}</span><span>{Math.round(model.size / 1024 / 1024) || 0} MB</span></div>
                </div>
                <div className="model-body"><div className="model-name">{model.name}</div><div className="model-type">CHECKPOINT <span>{model.source.toUpperCase()}</span></div><div className="model-tags">{model.tags.slice(0,4).map(t => <span key={t}>{t}</span>)}</div></div>
              </button>
            )}</div>}
        </section>

        <section className="panel lora-selector-panel">
          <div className="panel-head">
            <div><div className="kicker">COMPATIBLE LoRA LIBRARY</div><div className="panel-title">SELECT LoRAs MANUALLY</div></div>
            <div className="lora-count">{manualLoras.length} SELECTED · {compatibleLoras.length} COMPATIBLE</div>
          </div>
          {!selected ? <div className="lora-empty">SELECT A CHECKPOINT FIRST</div> :
           !compatibleLoras.length ? <div className="lora-empty">NO COMPATIBLE LoRAs FOUND FOR THIS CHECKPOINT</div> :
           <div className="lora-grid">
            {compatibleLoras.map(lora =>
              <button key={lora.id}
                className={'lora-card ' + (selectedLoraIds.includes(lora.id) ? 'selected' : '')}
                onClick={() => toggleLora(lora.id)}>
                <div className="lora-thumb">
                  {thumbs[lora.id] ? <img src={thumbs[lora.id]} alt="" /> : <div className="lora-fallback"><Layers3 size={18}/></div>}
                  {lora.character && <span>CHARACTER</span>}
                </div>
                <div className="lora-body">
                  <div className="lora-name">{lora.name}</div>
                  <div className="lora-tags">{lora.tags.slice(0,3).map(tag => <span key={tag}>{tag}</span>)}</div>
                </div>
                <div className="lora-check">{selectedLoraIds.includes(lora.id) ? <Check size={12}/> : ''}</div>
              </button>
            )}
           </div>}
        </section>

        <section className="lower-grid">
          <div className="panel">
            <div className="panel-head"><div><div className="kicker">USER DIRECTIVE</div><div className="panel-title">SCENE CONSTRAINTS</div></div><button className="ghost-btn" onClick={() => setConstraints(emptyConstraints)}><X size={13}/> CLEAR</button></div>
            <div className="field-grid">
              {(['setting','pose','expression','character','dress','composition'] as const).map(key =>
                <label key={key} className="wide-field"><span>{key.replace('_',' ').toUpperCase()}</span><input value={constraints[key]} onChange={e => updateConstraint(key,e.target.value)} placeholder={key === 'character' ? 'compatible character LoRA only' : 'blank = random'}/></label>
              )}
            </div>
            <label className="wide-field"><span>ADDITIONAL CONSTRAINTS</span><textarea value={constraints.additional} onChange={e => updateConstraint('additional',e.target.value)} placeholder="Anything the prompt engine must obey…"/></label>
            <div className="range-row"><label><span>RANDOM LoRA COUNT</span><div className="range-control"><input type="number" min={1} max={8} value={constraints.randomLoraMin} onChange={e => updateConstraint('randomLoraMin',Math.max(1,Number(e.target.value)))}/><b>TO</b><input type="number" min={constraints.randomLoraMin} max={8} value={constraints.randomLoraMax} onChange={e => updateConstraint('randomLoraMax',Math.max(constraints.randomLoraMin,Number(e.target.value)))}/></div></label></div>
            {prepared && <div className="stack-preview"><div className="stack-head"><span>SELECTED STACK</span><button onClick={() => void rollStack()}><RefreshCw size={12}/> ROLL AGAIN</button></div>{prepared.loras.map(l => <div className="stack-item" key={l.id}><span className={'stack-dot ' + (l.character ? 'character' : '')}/><div><b>{l.name}</b><small>{l.activationTags.join(', ')}</small></div><strong>{l.weight.toFixed(2)}</strong></div>)}</div>}
          </div>

          <div className="panel prompt-panel">
            <div className="panel-head"><div><div className="kicker">STREAMED MODEL OUTPUT</div><div className="panel-title">PROMPT ENGINE</div></div><span className="provider-chip">{provider === 'ollama' ? 'OLLAMA' : 'OPENAI COMPAT'}</span></div>
            <div className="stream-box">{stream ? <pre>{stream}</pre> : <div className="stream-placeholder"><Terminal size={18}/><span>LLM output will stream here token by token.</span></div>}</div>
            {prompts && <div className="prompt-result">
              <div className="prompt-block"><div className="prompt-block-head"><span>POSITIVE</span><button onClick={() => void copy(prompts.positive_prompt)}><Copy size={12}/> COPY</button></div><div className="prompt-text">{prompts.positive_prompt}</div></div>
              <div className="prompt-block"><div className="prompt-block-head"><span>NEGATIVE</span><button onClick={() => void copy(prompts.negative_prompt)}><Copy size={12}/> COPY</button></div><div className="prompt-text">{prompts.negative_prompt}</div></div>
            </div>}
            <div className="run-row"><button className="secondary-btn" disabled={busy || !selected} onClick={() => void rollStack()}><WandSparkles size={14}/> ROLL STACK</button><button className="primary-btn" disabled={busy || !selected || !llm.model} onClick={() => void generate()}><Play size={15}/> {busy ? 'GENERATING…' : 'GENERATE'}</button></div>
          </div>
        </section>
      </>}

      {tab === 'history' && <section className="history-panel">
        <div className="section-head"><div><div className="kicker">LOCAL RUN ARCHIVE</div><div className="section-title">GENERATION HISTORY</div></div><button onClick={() => void loadHistory()}><RefreshCw size={13}/> REFRESH</button></div>
        {!history.length ? <div className="empty-state"><History size={22}/><div><b>NO GENERATIONS RECORDED</b><span>Prompt, LoRA, workflow and ComfyUI records appear here.</span></div></div> :
          <div className="history-list">{history.map(item => <article className="history-card" key={item.id}><div className="history-main"><div className="history-meta"><span>{new Date(item.timestamp).toLocaleString()}</span><span>{item.model || '—'}</span><span>{item.checkpoint.name}</span></div><div className="history-scene">{item.scene.character} · {item.scene.setting} · {item.scene.pose} · {item.scene.expression}</div><div className="history-loras">{item.loras.map(l => <span key={l.id}>{l.name}</span>)}</div></div><button className="icon-btn" onClick={() => void copy(item.positivePrompt)}><Copy size={14}/></button></article>)}</div>}
      </section>}

      {tab === 'settings' && <section className="settings-grid">
        <div className="panel">
          <div className="panel-head"><div><div className="kicker">MODEL PROVIDER</div><div className="panel-title">LLM CONNECTION</div></div></div>
          <div className="provider-toggle"><button className={provider === 'ollama' ? 'active' : ''} onClick={() => {setProvider('ollama'); setLlm(x => ({...x, baseUrl:'http://127.0.0.1:11434'}));}}>OLLAMA</button><button className={provider === 'openai-compatible' ? 'active' : ''} onClick={() => {setProvider('openai-compatible'); setLlm(x => ({...x, baseUrl:'http://127.0.0.1:8080/v1'}));}}>OPENAI COMPATIBLE</button></div>
          <label className="wide-field"><span>BASE URL</span><input value={llm.baseUrl} onChange={e => setLlm(x => ({...x,baseUrl:e.target.value}))}/></label>
          <label className="wide-field"><span>API KEY / OPTIONAL</span><input type="password" value={llm.apiKey} onChange={e => setLlm(x => ({...x,apiKey:e.target.value}))}/></label>
          <div className="model-row"><label className="wide-field"><span>MODEL</span><select value={llm.model} onChange={e => setLlm(x => ({...x,model:e.target.value}))}><option value="">SELECT MODEL…</option>{models.map(model => <option key={model}>{model}</option>)}</select></label><button className="secondary-btn refresh-models" onClick={() => void refreshModels()}><RefreshCw size={14}/> GET MODELS</button></div>
          <div className="settings-ranges"><label className="wide-field"><span>TEMPERATURE</span><input type="range" min={0} max={1.5} step={0.05} value={llm.temperature} onChange={e => setLlm(x => ({...x,temperature:Number(e.target.value)}))}/></label><label className="wide-field"><span>MAX OUTPUT</span><input type="number" min={128} max={8192} step={128} value={llm.maxTokens} onChange={e => setLlm(x => ({...x,maxTokens:Number(e.target.value)}))}/></label></div>
        </div>
        <div className="panel">
          <div className="panel-head"><div><div className="kicker">GENERATION BACKEND</div><div className="panel-title">COMFYUI + RAPHAEL CACHE</div></div></div>
          <label className="wide-field"><span>COMFYUI API</span><input value={comfyUrl} onChange={e => setComfyUrl(e.target.value)}/></label>
          <label className="wide-field"><span>COMFYUI MODELS ROOT</span><div className="input-button"><input value={comfyRoot} onChange={e => setComfyRoot(e.target.value)}/><button onClick={() => void pickFolder(setComfyRoot)}><FolderOpen size={13}/></button></div></label>
          <label className="wide-field"><span>RAPHAEL CACHE / MANAGER ROOT</span><div className="input-button"><input value={raphaelRoot} onChange={e => setRaphaelRoot(e.target.value)}/><button onClick={() => void pickFolder(setRaphaelRoot)}><FolderOpen size={13}/></button></div></label>
          <button className="primary-btn full" onClick={() => void scan()}><RefreshCw size={14}/> SCAN + MERGE CACHE</button>
        </div>
      </section>}

      {error && <div className="error-box"><CircleAlert size={14}/><span>{error}</span></div>}
      {toast && <div className="toast">{toast}</div>}
    </main>

    <aside className="right-rail">
      <div className="rail-label">ACTIVE CHECKPOINT</div>
      {selected ? <div className="active-model"><div className="active-model-thumb">{thumbs[selected.id] ? <img src={thumbs[selected.id]} alt=""/> : <Layers3 size={24}/>}</div><div className="active-name">{selected.name}</div><div className="active-sub">{selected.baseModel || 'BASE NOT RESOLVED'}</div><div className="active-tags">{selected.tags.slice(0,7).map(t => <span key={t}>{t}</span>)}</div></div> : <div className="active-empty">SELECT A CHECKPOINT</div>}
      <div className="rail-divider"/><div className="rail-label">LIVE STATUS</div>
      <div className="stage-stack">{stages.map(s => <div className="mini-stage" key={s.key}><span className={'status ' + status[s.key]}/><span>{s.label}</span><b>{status[s.key].toUpperCase()}</b></div>)}</div>
    </aside>
  </div>;
}

export default App;
