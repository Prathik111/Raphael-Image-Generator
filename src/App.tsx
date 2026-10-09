import { useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { Channel, invoke } from '@tauri-apps/api/core';
import {
  Check, CircleAlert, Copy, Globe2, History, Layers3, Play,
  RefreshCw, Search, Settings2, Sparkles, Terminal, WandSparkles, X
} from 'lucide-react';
import type {
  Constraints, DemographicLevel, DemographicPrompts, GenerationRecord, GenerationSettings, LibrarySnapshot,
  LlmSettings, PreparedGeneration, PromptPair, ProviderKind, SceneSelection, WebHostInfo
} from './types';

type Stage = 'library' | 'compatibility' | 'selection' | 'planning' | 'llm' | 'workflow' | 'comfy' | 'recorded';
type Status = 'idle' | 'running' | 'done' | 'error';

interface CharacterPlan {
  name: string;
  gender: string;
  appearance: string;
  pose: string;
  expression: string;
  position: string;
  interaction: string;
}

type PlannerAnchorType = 'concept' | 'action' | 'setting' | 'background' | 'pose' | 'composition' | 'lighting';

interface GenerationPlan {
  characters: CharacterPlan[];
  character: string;
  concept: string;
  anchorType: PlannerAnchorType;
  anchor: string;
  action: string;
  pose: string;
  setting: string;
  background: string;
  expression: string;
  dress: string;
  composition: string;
  lighting: string;
  camera: string;
  framing: string;
}

interface PromptValidation {
  valid: boolean;
  errors: string[];
  positiveTags: string[];
  negativeTags: string[];
}

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

const defaultPlannerSystemPrompt = `You are the VISUAL PLANNER for an image-generation pipeline.

Your job is to decide the concrete visual content of an image before another model converts that decision into image-model tags.

Do not generate the final positive or negative prompt.
Do not write prose, narrative descriptions, metaphors, or image-model prompt tags.

Return JSON only with exactly these fields:
characters, concept, action, pose, setting, background, expression, dress, composition, lighting, camera, framing.

"characters" must be an array with exactly one object for EVERY selected CHARACTER IDENTITY LoRA.

SCENE-COHERENCE METHOD:
The application supplies one SCENE ANCHOR CATEGORY. Use that category as the primary creative decision.
First choose one concrete anchor value for that category. Then derive every other scene field so it naturally fits that anchor.
Do NOT choose pose, setting, background, expression, lighting, clothing, composition, camera, or framing independently.
Every downstream choice must support the same moment, place, action, mood, time, weather, and visual logic.
If the anchor is a pose, choose an environment/action that makes that pose plausible.
If the anchor is a background, choose setting, lighting, action, pose, composition, and camera that make that background plausible.
If the anchor is an action, choose character placement, pose, expression, setting, background, composition, lighting, and camera that support that action.
If the anchor is a concept, treat the concept as the scene's central idea and make every field a consequence of it.
Explicit user constraints always override the random anchor category; adapt the rest of the scene around those constraints.
Never produce a collection of unrelated "random" choices. The result must read as one coherent photograph or illustration taken at one specific moment.

If 2 character LoRAs are selected, return exactly 2 character objects.
Each character object must contain:
name, gender, appearance, pose, expression, position, interaction.

If multiple character LoRAs are selected, NEVER merge them into one character.
If 2 character LoRAs are selected, return exactly 2 character objects.
If 3 character LoRAs are selected, return exactly 3 character objects.
The "gender" field must state the known/established gender of the character when available. Use concise values such as "female", "male", "non-binary", or "unknown"; never invent an unsupported gender.

Preserve the identity and documented purpose of every selected character LoRA.

Use short canonical visual choices, preferably 1-6 words per field.
Make one concrete decision for every field.
Preserve explicit user constraints.
When a field is blank or random, choose a specific value that fits the rest of the scene.
Keep action, pose, expression, setting, background, composition, lighting, camera, and framing internally consistent.
Do not mention filenames, implementation details, or LoRA syntax.

CONTENT POLICY:
{{DEMOGRAPHIC_POLICY}}`;

const defaultTagSystemPrompt = `You are the TAG GENERATION MODEL in an image-generation pipeline.

Convert the supplied scene and locked visual plan into a deterministic image-model prompt pair.

Return JSON only:
{
  "positive_prompt": "",
  "negative_prompt": ""
}

The positive prompt must contain at least {{MIN_POSITIVE_TAGS}} meaningful comma-separated visual tags or short phrases and target approximately 90-160 words.
Use literal canonical visual terms, mostly 1-6 words per tag and no more than {{MAX_TAG_LENGTH}} characters per ordinary tag.
Do not write sentences, narrative prose, metaphors, storytelling, or poetic language.
Do not invent a different pose, action, background, expression, lighting, camera, framing, or composition from the locked plan.

Cover the subject identity and visible appearance, current action/state, pose, hands and arms when relevant, head direction, gaze, facial expression, clothing and accessories, interaction, setting, background, atmosphere, composition, perspective, depth, lighting, camera viewpoint, framing, materials, and useful finishing details.

The negative prompt must contain between {{MIN_NEGATIVE_TAGS}} and {{MAX_NEGATIVE_TAGS}} concrete comma-separated failure tags targeted to the image.
Do not mention checkpoint names, filenames, model implementation syntax, or LoRA syntax.

Every documented activation prompt supplied in the user message must be preserved exactly, unchanged, at least once, and placed next to the visual concept it controls.
Never invent, paraphrase, abbreviate, or move words inside an activation prompt.

CONTENT POLICY:
{{DEMOGRAPHIC_POLICY}}`;

const defaultRepairSystemPrompt = `You are the TAG REPAIRER in an image-generation pipeline.

The previous tag-generation result failed validation. Rewrite it into a valid final prompt pair while preserving the locked visual plan and required activation prompts.

Return JSON only:
{
  "positive_prompt": "",
  "negative_prompt": ""
}

Fix every validation problem explicitly reported in the user message.
The positive prompt must contain at least {{MIN_POSITIVE_TAGS}} meaningful comma-separated visual tags and target approximately 90-160 words.
The negative prompt must contain between {{MIN_NEGATIVE_TAGS}} and {{MAX_NEGATIVE_TAGS}} concrete comma-separated failure tags.
Ordinary tags must be 1-6 words and no more than {{MAX_TAG_LENGTH}} characters.
Do not use sentence-like tags, prose, metaphors, narrative clauses, or explanations.

Do not change the planned character, action, pose, setting, background, expression, dress, composition, lighting, camera, or framing unless the validation error explicitly requires repair of an invalid value.

Every documented activation prompt must be preserved exactly as provided and must appear at least once.
Do not paraphrase or modify activation prompts.
Do not output LoRA implementation syntax, checkpoint names, filenames, or model names.

Use the previous JSON only as the material to repair. Do not blindly append random tags just to meet minimum counts.

CONTENT POLICY:
{{DEMOGRAPHIC_POLICY}}`;

const defaultPlanningPrompt = `GENERATION INPUT FOR THE PLANNER

CHECKPOINT: {{CHECKPOINT}}
BASE: {{BASE}}

SCENE ANCHOR CATEGORY: {{SCENE_ANCHOR_TYPE}}
The planner must choose one concrete anchor value in this category first.
All other scene fields must be derived from that single anchor.

SELECTED LoRA METADATA:
{{LORA_METADATA}}

SELECTED CHARACTER IDENTITY LoRAs:
{{CHARACTERS}}

OUTPUT REQUIREMENT:
The JSON MUST contain a "characters" array with exactly {{CHARACTER_COUNT}} objects, one for each selected character LoRA.
Never put multiple character names into a single "character" string.

USER CONSTRAINTS:
CHARACTER: {{CHARACTER}}
SETTING: {{SETTING}}
POSE: {{POSE}}
EXPRESSION: {{EXPRESSION}}
DRESS: {{DRESS}}
COMPOSITION: {{COMPOSITION}}
EXTRA: {{EXTRA}}`;

const defaultUserPromptTemplate = `GENERATION INPUT FOR TAG CREATION

CHECKPOINT: {{CHECKPOINT}}
BASE: {{BASE}}
COMPATIBILITY: {{COMPATIBILITY}}

SELECTED LoRAs AND THEIR DOCUMENTED PURPOSE METADATA:
{{LORA_METADATA}}

SCENE:
CHARACTER: {{CHARACTER}}
SETTING: {{SETTING}}
POSE: {{POSE}}
EXPRESSION: {{EXPRESSION}}
DRESS: {{DRESS}}
COMPOSITION: {{COMPOSITION}}
EXTRA: {{EXTRA}}

LOCKED PLAN:
The application will append the planner's locked visual decisions after this context.`;

const defaultExpansionPromptTemplate = `GENERATION INPUT FOR TAG REPAIR

CHECKPOINT: {{CHECKPOINT}}
BASE: {{BASE}}
COMPATIBILITY: {{COMPATIBILITY}}

SCENE:
CHARACTER: {{CHARACTER}}
SETTING: {{SETTING}}
POSE: {{POSE}}
EXPRESSION: {{EXPRESSION}}
DRESS: {{DRESS}}
COMPOSITION: {{COMPOSITION}}
EXTRA: {{EXTRA}}

SELECTED LoRA METADATA:
{{LORA_METADATA}}

PREVIOUS JSON:
{{PREVIOUS_JSON}}`;

const generationSettingsStorageKey = 'raphael-image-generator.generation-settings.v2';
const defaultMinPositiveTags = 24;
const defaultMinNegativeTags = 20;
const defaultMaxNegativeTags = 35;
const absoluteMaxTagLimit = 100;
const defaultMaxTagLength = 64;
const absoluteMaxTagLength = 256;
const defaultPlannerTemperature = 0.35;
const defaultTagTemperature = 0.72;
const defaultTagGenerationRetries = 2;
const absoluteMaxTagGenerationRetries = 8;
const defaultMaxCharacterLoras = 1;
const absoluteMaxLoraLimit = 16;

interface StoredGenerationSettings {
  llm?: LlmSettings;
  demographic?: DemographicLevel;
  demographicPrompts?: DemographicPrompts;
  maxLoras?: number;
  randomLoraMin?: number;
  randomLoraMax?: number;
  constraints?: Constraints;
  width?: number;
  height?: number;
  steps?: number;
  cfg?: number;
  sampler?: string;
  userPromptTemplate?: string;
  expansionPromptTemplate?: string;
  plannerSystemPrompt?: string;
  tagSystemPrompt?: string;
  repairSystemPrompt?: string;
  minPositiveTags?: number;
  minNegativeTags?: number;
  maxNegativeTags?: number;
  maxTagLength?: number;
  plannerTemperature?: number;
  tagTemperature?: number;
  tagGenerationRetries?: number;
  maxCharacterLoras?: number;
  manualLoraIds?: string[];
}

function loadPersistedGenerationSettings(): StoredGenerationSettings {
  if(typeof window==='undefined') return {};
  try{
    const raw=window.localStorage.getItem(generationSettingsStorageKey);
    if(!raw) return {};
    const parsed=JSON.parse(raw) as StoredGenerationSettings;
    if(!parsed || typeof parsed!=='object') return {};

    const migrated:StoredGenerationSettings={...parsed};

    // Older v2 settings stored behavioral rules inside the user templates.
    // Move only those known legacy defaults so custom user templates survive.
    if(typeof migrated.userPromptTemplate==='string'
      && /Write the FINAL positive and negative prompts/i.test(migrated.userPromptTemplate)){
      migrated.userPromptTemplate=defaultUserPromptTemplate;
    }
    if(typeof migrated.expansionPromptTemplate==='string'
      && /EXPANSION PASS\. Rewrite the previous result/i.test(migrated.expansionPromptTemplate)){
      migrated.expansionPromptTemplate=defaultExpansionPromptTemplate;
    }

    const persistedPlannerPrompt =
      typeof migrated.plannerSystemPrompt==='string'
        ? migrated.plannerSystemPrompt.trim()
        : '';

    migrated.plannerSystemPrompt =
      !persistedPlannerPrompt
        || /Return JSON only with exactly these string fields:\s*character, action, pose/i.test(persistedPlannerPrompt)
        ? defaultPlannerSystemPrompt
        : persistedPlannerPrompt.includes('SCENE-COHERENCE METHOD:')
          ? persistedPlannerPrompt.includes('gender, appearance, pose')
            ? persistedPlannerPrompt
            : persistedPlannerPrompt + '\n\nCHARACTER SCHEMA UPDATE (required): Each character object must contain name, gender, appearance, pose, expression, position, interaction. Never omit gender; use a known concise value or "unknown".'
          : persistedPlannerPrompt
            + '\n\nSCENE-COHERENCE METHOD (required): Choose one concrete scene anchor first, using the SCENE ANCHOR CATEGORY supplied by the application. Derive every other scene field from that anchor. Do not independently randomize pose, setting, background, expression, lighting, clothing, composition, camera, or framing. Every field must describe one coherent moment. Explicit user constraints override the anchor category; adapt the rest of the scene around them.';
    migrated.tagSystemPrompt =
      typeof migrated.tagSystemPrompt==='string' && migrated.tagSystemPrompt.trim()
        ? migrated.tagSystemPrompt
        : defaultTagSystemPrompt;
    migrated.repairSystemPrompt =
      typeof migrated.repairSystemPrompt==='string' && migrated.repairSystemPrompt.trim()
        ? migrated.repairSystemPrompt
        : defaultRepairSystemPrompt;

    if(typeof migrated.tagSystemPrompt==='string'
      && migrated.tagSystemPrompt.includes('You are the TAG GENERATION MODEL')){
      migrated.tagSystemPrompt=migrated.tagSystemPrompt
        .replace(/at least 24 meaningful/gi,'at least {{MIN_POSITIVE_TAGS}} meaningful')
        .replace(/20-35 concrete/gi,'between {{MIN_NEGATIVE_TAGS}} and {{MAX_NEGATIVE_TAGS}} concrete')
        .replace(/between \{\{MIN_NEGATIVE_TAGS\}\} and 35 concrete/gi,'between {{MIN_NEGATIVE_TAGS}} and {{MAX_NEGATIVE_TAGS}} concrete')
        .replace(/no more than 64 characters(?: per ordinary tag)?/gi,'no more than {{MAX_TAG_LENGTH}} characters per ordinary tag');
    }
    if(typeof migrated.repairSystemPrompt==='string'
      && migrated.repairSystemPrompt.includes('You are the TAG REPAIRER')){
      migrated.repairSystemPrompt=migrated.repairSystemPrompt
        .replace(/at least 24 meaningful/gi,'at least {{MIN_POSITIVE_TAGS}} meaningful')
        .replace(/20-35 concrete/gi,'between {{MIN_NEGATIVE_TAGS}} and {{MAX_NEGATIVE_TAGS}} concrete')
        .replace(/between \{\{MIN_NEGATIVE_TAGS\}\} and 35 concrete/gi,'between {{MIN_NEGATIVE_TAGS}} and {{MAX_NEGATIVE_TAGS}} concrete')
        .replace(/no more than 64 characters/gi,'no more than {{MAX_TAG_LENGTH}} characters');
    }

    migrated.maxNegativeTags=Math.max(1,Math.min(absoluteMaxTagLimit,migrated.maxNegativeTags ?? defaultMaxNegativeTags));
    migrated.maxTagLength=Math.max(1,Math.min(absoluteMaxTagLength,migrated.maxTagLength ?? defaultMaxTagLength));
    migrated.tagGenerationRetries=Math.max(0,Math.min(absoluteMaxTagGenerationRetries,migrated.tagGenerationRetries ?? defaultTagGenerationRetries));
    migrated.minPositiveTags=Math.max(1,Math.min(100,migrated.minPositiveTags ?? defaultMinPositiveTags));
    migrated.minNegativeTags=Math.max(1,Math.min(migrated.maxNegativeTags,migrated.minNegativeTags ?? defaultMinNegativeTags));

    return migrated;
  }catch{
    return {};
  }
}

function renderPromptTemplate(
  template:string,
  values:Record<string,string>,
){
  return template.replace(/\{\{([A-Z0-9_]+)\}\}/g,(match,key)=>
    Object.prototype.hasOwnProperty.call(values,key) ? values[key] : match
  );
}


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
  {key:'planning',label:'PLANNING'},
  {key:'llm',label:'TAG GENERATION'},
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
function extractJsonObject(raw:string):Record<string,unknown>{
  let clean=raw.trim();
  if(clean.includes('</think>')) clean=clean.slice(clean.lastIndexOf('</think>')+8).trim();
  clean=clean.replace(/```json/gi,'').replace(/```/g,'').trim();
  const start=clean.indexOf('{');
  const end=clean.lastIndexOf('}');
  if(start<0 || end<start) throw new Error('LLM did not return a JSON object.');
  const value=JSON.parse(clean.slice(start,end+1)) as unknown;
  if(!value || typeof value!=='object' || Array.isArray(value)) throw new Error('LLM returned a non-object JSON value.');
  return value as Record<string,unknown>;
}

interface SelectedCharacterInput {
  name:string;
  description?:string|null;
  activationTags:string[];
}

function textField(value:unknown,fallback=''):string{
  return typeof value==='string' ? value.trim() : fallback;
}

function pickPlannerAnchorType(constraints:Constraints):PlannerAnchorType{
  const candidates:Array<{type:PlannerAnchorType;locked:boolean}>= [
    {type:'concept',locked:false},
    {type:'action',locked:false},
    {type:'setting',locked:Boolean(constraints.setting.trim())},
    {type:'background',locked:false},
    {type:'pose',locked:Boolean(constraints.pose.trim())},
    {type:'composition',locked:Boolean(constraints.composition.trim())},
    {type:'lighting',locked:false},
  ];
  const open=candidates.filter(candidate=>!candidate.locked);
  const pool=open.length ? open : candidates;
  return pool[Math.floor(Math.random()*pool.length)].type;
}

function parseGenerationPlan(
  raw:string,
  fallback:SceneSelection,
  expectedCharacterLoras:SelectedCharacterInput[],
  anchorType:PlannerAnchorType,
):GenerationPlan{
  const value=extractJsonObject(raw);
  const rawCharacters=Array.isArray(value.characters)
    ? value.characters
    : [];

  // Some locally persisted/older planner prompts still emit the legacy
  // single "character" string. Normalize both legacy and current outputs
  // into exactly one plan record per selected character LoRA.
  const legacyCharacterNames=textField(value.character)
    .split(/[\\n,|]+/)
    .map(name=>name.trim())
    .filter(Boolean);

  const plannerItems:Record<string,unknown>[]=rawCharacters
    .filter(item=>item && typeof item==='object')
    .map(item=>item as Record<string,unknown>);

  const characters:CharacterPlan[]=expectedCharacterLoras.map((fallbackLora,index)=>{
    const obj=plannerItems[index] || {};
    const legacyName=legacyCharacterNames[index];

    return {
      name:textField(
        obj.name,
        legacyName || fallbackLora.name || 'Character '+(index+1),
      ),
      gender:textField(
        obj.gender,
        'unknown',
      ),
      appearance:textField(
        obj.appearance,
        fallbackLora.description || fallbackLora.name || 'distinct character appearance',
      ),
      pose:textField(
        obj.pose,
        textField(value.pose,'natural pose'),
      ),
      expression:textField(
        obj.expression,
        textField(value.expression,'calm expression'),
      ),
      position:textField(
        obj.position,
        index===0 ? 'primary position' : 'secondary position',
      ),
      interaction:textField(
        obj.interaction,
        index===0 ? 'leading the scene interaction' : 'interacting with the other characters',
      ),
    };
  });
  const character=textField(value.character,
    characters.map((item,index)=>'CHARACTER '+(index+1)+': '+item.name+' — '+item.appearance).join('\n')
      || fallback.character
  );
  const planFields={
    characters,
    character,
    concept:textField(value.concept,'coherent visual concept'),
    action:textField(value.action,'coherent action'),
    pose:textField(value.pose,fallback.pose || 'natural pose'),
    setting:textField(value.setting,fallback.setting || 'coherent setting'),
    background:textField(value.background,'coherent background'),
    expression:textField(value.expression,fallback.expression || 'natural expression'),
    dress:textField(value.dress,fallback.dress || 'coherent outfit'),
    composition:textField(value.composition,fallback.composition || 'balanced composition'),
    lighting:textField(value.lighting,'lighting consistent with the anchor'),
    camera:textField(value.camera,'camera suited to the anchor'),
    framing:textField(value.framing,fallback.composition || 'framing suited to the anchor'),
  };
  const anchorValues:Record<PlannerAnchorType,string>={
    concept:planFields.concept,
    action:planFields.action,
    setting:planFields.setting,
    background:planFields.background,
    pose:planFields.pose,
    composition:planFields.composition,
    lighting:planFields.lighting,
  };
  const plan:GenerationPlan={
    ...planFields,
    anchorType,
    anchor:anchorValues[anchorType],
  };
  const missing=Object.entries(plan).filter(([,v])=>typeof v==='string' && !v.trim()).map(([k])=>k);
  if(missing.length) throw new Error('Planning response is missing: '+missing.join(', '));
  return plan;
}

function splitPromptTags(value:string):string[]{
  return value.split(',').map(tag=>tag.trim()).filter(Boolean);
}

function validateGeneratedPrompt(
  pair:PromptPair,
  activationTags:string[],
  minPositiveTags:number,
  minNegativeTags:number,
  maxNegativeTags:number,
  maxTagLength:number,
):PromptValidation{
  const positiveTags=splitPromptTags(pair.positive_prompt);
  const negativeTags=splitPromptTags(pair.negative_prompt);
  const exactActivationTags=new Set(activationTags.map(tag=>tag.trim()).filter(Boolean));
  const errors:string[]=[];

  if(positiveTags.length<minPositiveTags) errors.push(`positive prompt has only ${positiveTags.length} tags; minimum is ${minPositiveTags}`);
  if(negativeTags.length<minNegativeTags) errors.push(`negative prompt has only ${negativeTags.length} tags; minimum is ${minNegativeTags}`);
  if(negativeTags.length>maxNegativeTags) errors.push(`negative prompt has ${negativeTags.length} tags; maximum is ${maxNegativeTags}`);

  const validateTagList=(tags:string[],label:string,allowExactActivation:boolean)=>{
    tags.forEach((tag,index)=>{
      const words=tag.split(/\s+/).filter(Boolean).length;
      const chars=tag.length;
      const exactActivation=allowExactActivation && exactActivationTags.has(tag);
      const sentenceLike=/[.!?;:]$/.test(tag) || /\b(because|therefore|while|which|that|this|these|then|and then)\b/i.test(tag);
      if(!exactActivation && (words>6 || chars>maxTagLength || sentenceLike)){
        errors.push(`${label} tag ${index+1} is sentence-like, over ${maxTagLength} characters, or exceeds 6 words: "${tag}"`);
      }
    });
  };

  validateTagList(positiveTags,'positive',true);
  validateTagList(negativeTags,'negative',false);
  return {valid:errors.length===0,errors,positiveTags,negativeTags};
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
  const persistedGenerationSettings=loadPersistedGenerationSettings();
  const initialLlm: LlmSettings={
    provider:'ollama',
    baseUrl:'http://127.0.0.1:11434',
    apiKey:'',
    model:'',
    temperature:0.72,
    maxTokens:8192,
    contextTokens:32768,
    ...(persistedGenerationSettings.llm || {}),
  };
  const [provider,setProvider]=useState<ProviderKind>(initialLlm.provider);
  const [llm,setLlm]=useState<LlmSettings>(initialLlm);
  const [models,setModels]=useState<string[]>([]);
  const [library,setLibrary]=useState<LibrarySnapshot|null>(null);
  const [selectedId,setSelectedId]=useState('');
  const [comfyRoot,setComfyRoot]=useState('D:\\ComfyUI\\models');
  const [registryUrl,setRegistryUrl]=useState('');
  const [comfyUrl,setComfyUrl]=useState('http://127.0.0.1:8188');

  const [constraints,setConstraints]=useState<Constraints>(persistedGenerationSettings.constraints || emptyConstraints);
  const [demographic,setDemographic]=useState<DemographicLevel>(persistedGenerationSettings.demographic || 'safe');
  const [demographicPrompts,setDemographicPrompts]=useState<DemographicPrompts>(persistedGenerationSettings.demographicPrompts || loadPersistedDemographicPrompts);
  const [maxLoras,setMaxLoras]=useState(persistedGenerationSettings.maxLoras || 4);
  const [width,setWidth]=useState(persistedGenerationSettings.width || 1024);
  const [height,setHeight]=useState(persistedGenerationSettings.height || 1024);
  const [steps,setSteps]=useState(persistedGenerationSettings.steps || 28);
  const [cfg,setCfg]=useState(persistedGenerationSettings.cfg ?? 6.5);
  const [sampler,setSampler]=useState(persistedGenerationSettings.sampler || 'euler');
  const [minPositiveTags,setMinPositiveTags]=useState(
    Math.max(1,Math.min(100,persistedGenerationSettings.minPositiveTags ?? defaultMinPositiveTags)),
  );
  const [minNegativeTags,setMinNegativeTags]=useState(
    Math.max(1,Math.min(persistedGenerationSettings.maxNegativeTags ?? defaultMaxNegativeTags,persistedGenerationSettings.minNegativeTags ?? defaultMinNegativeTags)),
  );
  const [prepared,setPrepared]=useState<PreparedGeneration|null>(null);
  const [prompts,setPrompts]=useState<PromptPair|null>(null);
  const [stream,setStream]=useState('');
  const [history,setHistory]=useState<GenerationRecord[]>([]);
  const [tab,setTab]=useState<'generate'|'history'>('generate');

  const [stage,setStage]=useState<Stage>('library');
  const [status,setStatus]=useState<Record<Stage,Status>>({
    library:'idle', compatibility:'idle', selection:'idle', planning:'idle', llm:'idle',
    workflow:'idle', comfy:'idle', recorded:'idle',
  });
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const [toast,setToast]=useState('');
  const [plan,setPlan]=useState<GenerationPlan|null>(null);
  const [planStream,setPlanStream]=useState('');
  const [tagValidation,setTagValidation]=useState('');
  const persistedManualLoraIds=Array.isArray(persistedGenerationSettings.manualLoraIds)
    ? persistedGenerationSettings.manualLoraIds.filter(id=>typeof id==='string')
    : [];
  const [selectedLoraIds,setSelectedLoraIds]=useState<string[]>(persistedManualLoraIds);
  const [manualLoraIds,setManualLoraIds]=useState<string[]>(persistedManualLoraIds);

  const [settingsOpen,setSettingsOpen]=useState(false);
  const [mobileControlsOpen,setMobileControlsOpen]=useState(false);
  const [settingsTab,setSettingsTab]=useState<'llm'|'system-prompts'|'prompt-templates'|'scene'|'output'>('llm');
  const [generationDraft,setGenerationDraft]=useState({
    llm,
    demographic,
    demographicPrompts,
    maxLoras,
    randomLoraMin:constraints.randomLoraMin || persistedGenerationSettings.randomLoraMin || 2,
    randomLoraMax:constraints.randomLoraMax || persistedGenerationSettings.randomLoraMax || 4,
    constraints,
    width,
    height,
    steps,
    cfg,
    sampler,
    plannerSystemPrompt:persistedGenerationSettings.plannerSystemPrompt || defaultPlannerSystemPrompt,
    tagSystemPrompt:persistedGenerationSettings.tagSystemPrompt || defaultTagSystemPrompt,
    repairSystemPrompt:persistedGenerationSettings.repairSystemPrompt || defaultRepairSystemPrompt,
    userPromptTemplate:persistedGenerationSettings.userPromptTemplate || defaultUserPromptTemplate,
    expansionPromptTemplate:persistedGenerationSettings.expansionPromptTemplate || defaultExpansionPromptTemplate,
    minPositiveTags:Math.max(1,Math.min(100,persistedGenerationSettings.minPositiveTags ?? defaultMinPositiveTags)),
    minNegativeTags:Math.max(1,Math.min(persistedGenerationSettings.maxNegativeTags ?? defaultMaxNegativeTags,persistedGenerationSettings.minNegativeTags ?? defaultMinNegativeTags)),
    maxNegativeTags:Math.max(1,Math.min(absoluteMaxTagLimit,persistedGenerationSettings.maxNegativeTags ?? defaultMaxNegativeTags)),
    maxTagLength:Math.max(1,Math.min(absoluteMaxTagLength,persistedGenerationSettings.maxTagLength ?? defaultMaxTagLength)),
    plannerTemperature:Math.max(0,Math.min(2,persistedGenerationSettings.plannerTemperature ?? defaultPlannerTemperature)),
    tagTemperature:Math.max(0,Math.min(2,persistedGenerationSettings.tagTemperature ?? defaultTagTemperature)),
    tagGenerationRetries:Math.max(0,Math.min(absoluteMaxTagGenerationRetries,persistedGenerationSettings.tagGenerationRetries ?? defaultTagGenerationRetries)),
    maxCharacterLoras:Math.max(1,Math.min(absoluteMaxLoraLimit,persistedGenerationSettings.maxCharacterLoras ?? defaultMaxCharacterLoras)),
  });
  const [selectedHistoryId,setSelectedHistoryId]=useState('');
  const [historySettingsVisible,setHistorySettingsVisible]=useState(false);
  const [checkpointSearch,setCheckpointSearch]=useState('');
  const [loraSearch,setLoraSearch]=useState('');

  const [webHost,setWebHost]=useState<WebHostInfo|null>(null);
  const [lanClientReady,setLanClientReady]=useState(false);
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
  // Protect in-progress LAN edits from an older host-settings poll response.
  const generationSettingsEditedAt=useRef(0);
  // Poll the tiny history revision marker; do not repeatedly transfer and parse
  // the whole archive (which can contain hundreds of MiB of base64 image data).
  const historyRevisionRef=useRef('');
  const historyLoadInFlight=useRef(false);

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

  const selectedCharacterLoras=useMemo(
    ()=>allLoras.filter(lora=>
      selectedLoraIds.includes(lora.id) &&
      (lora.character || lora.tags.some(tag=>normUi(tag)==='character'))
    ),
    [allLoras,selectedLoraIds],
  );

  const selectedCharacterLoraCount=selectedCharacterLoras.length;

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

  const duplicateCheckpointNames=useMemo(()=>{
    const counts=new Map<string,number>();
    for(const model of library?.checkpoints || []){
      const key=model.name.trim().toLowerCase();
      counts.set(key,(counts.get(key) || 0)+1);
    }
    return counts;
  },[library]);

  const duplicateLoraNames=useMemo(()=>{
    const counts=new Map<string,number>();
    for(const model of allLoras){
      const key=model.name.trim().toLowerCase();
      counts.set(key,(counts.get(key) || 0)+1);
    }
    return counts;
  },[allLoras]);

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
        const hostConfig=await apiInvoke<{
          provider:ProviderKind;
          baseUrl:string;
          model:string;
          temperature?:number;
          maxTokens?:number;
          contextTokens?:number;
          generationSettings?:Partial<StoredGenerationSettings>;
        }>('get_host_llm_config');
        const hostGeneration=hostConfig.generationSettings || {};
        const nextLlm={
          ...llm,
          provider:hostConfig.provider,
          baseUrl:hostConfig.baseUrl,
          model:hostConfig.model || '',
          temperature:hostConfig.temperature ?? llm.temperature,
          maxTokens:hostConfig.maxTokens ?? llm.maxTokens,
          contextTokens:hostConfig.contextTokens ?? llm.contextTokens,
        };
        setLlm(nextLlm);
        setGenerationDraft(d=>({
          ...d,
          ...hostGeneration,
          llm:{
            ...d.llm,
            ...(hostGeneration.llm || {}),
            provider:hostConfig.provider,
            baseUrl:hostConfig.baseUrl,
            model:hostConfig.model || '',
            temperature:hostConfig.temperature ?? d.llm.temperature,
            maxTokens:hostConfig.maxTokens ?? d.llm.maxTokens,
            contextTokens:hostConfig.contextTokens ?? d.llm.contextTokens,
          },
        }));
        await fetchModels(nextLlm);
        setLanClientReady(true);
      }

      await scan(config.models_root || comfyRoot,config.registry_url || undefined);
    }catch(e){
      setError(String(e));
    }
  }

  async function loadHistory(showError=false){
    if(historyLoadInFlight.current) return;
    historyLoadInFlight.current=true;
    try{
      // The history contains inline base64 images, so fetching it every polling
      // interval repeatedly moves/parses a potentially very large JSON document.
      // A tiny file revision tells us whether a full read is actually required.
      const revision=await apiInvoke<string>('history_revision');
      if(revision===historyRevisionRef.current) return;

      const records=await apiInvoke<Array<{payload:GenerationRecord}>>('load_history');
      if(!Array.isArray(records)) throw new Error('History response was not a list.');
      setHistory(records.map(x=>x.payload).filter(Boolean));
      historyRevisionRef.current=revision;
    }catch(e){
      if(showError) setError('Could not load host generation history: '+String(e));
    }finally{
      historyLoadInFlight.current=false;
    }
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

      // Keep the selected Registry-file identity when it still exists.
      // When a Manager deletion removes that file, immediately move selection
      // to the first remaining checkpoint instead of retaining a dead ID and
      // silently falling back to an unrelated model.
      setSelectedId(current=>{
        if(current && snap.checkpoints.some(model=>model.id===current)) return current;
        return snap.checkpoints[0]?.id || '';
      });

      const ids=new Set(snap.loras.map(x=>x.id));
      setManualLoraIds(current=>current.filter(id=>ids.has(id)));
      setSelectedLoraIds(current=>current.filter(id=>ids.has(id)));
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

  useEffect(()=>{
    try{
      window.localStorage.setItem(
        generationSettingsStorageKey,
        JSON.stringify({
          ...loadPersistedGenerationSettings(),
          manualLoraIds,
        }),
      );
    }catch{}
  },[manualLoraIds]);

  useEffect(()=>{
    const draft=generationDraft;
    const cappedMax=Math.max(1,Math.min(absoluteMaxLoraLimit,draft.maxLoras));
    const nextConstraints={
      ...draft.constraints,
      randomLoraMin:Math.max(1,Math.min(cappedMax,draft.randomLoraMin)),
      randomLoraMax:Math.max(1,Math.min(cappedMax,draft.randomLoraMax)),
    };
    const normalizedDraft={
      ...draft,
      maxLoras:cappedMax,
      maxNegativeTags:Math.max(1,Math.min(absoluteMaxTagLimit,Number(draft.maxNegativeTags) || defaultMaxNegativeTags)),
      maxTagLength:Math.max(1,Math.min(absoluteMaxTagLength,Number(draft.maxTagLength) || defaultMaxTagLength)),
      plannerTemperature:Math.max(0,Math.min(2,Number(draft.plannerTemperature) || 0)),
      tagTemperature:Math.max(0,Math.min(2,Number(draft.tagTemperature) || 0)),
      tagGenerationRetries:Math.max(0,Math.min(absoluteMaxTagGenerationRetries,Number.isFinite(Number(draft.tagGenerationRetries)) ? Number(draft.tagGenerationRetries) : defaultTagGenerationRetries)),
      maxCharacterLoras:Math.max(1,Math.min(cappedMax,Number(draft.maxCharacterLoras) || defaultMaxCharacterLoras)),
      minPositiveTags:Math.max(1,Math.min(100,Number(draft.minPositiveTags) || defaultMinPositiveTags)),
      minNegativeTags:Math.max(1,Math.min(Math.max(1,Math.min(absoluteMaxTagLimit,Number(draft.maxNegativeTags) || defaultMaxNegativeTags)),Number(draft.minNegativeTags) || defaultMinNegativeTags)),
      randomLoraMin:nextConstraints.randomLoraMin,
      randomLoraMax:nextConstraints.randomLoraMax,
      constraints:nextConstraints,
      width:Math.max(64,draft.width),
      height:Math.max(64,draft.height),
      steps:Math.max(1,draft.steps),
      cfg:Math.max(0,draft.cfg),
      sampler:draft.sampler || 'euler',
    };
    setProvider(normalizedDraft.llm.provider);
    setLlm({...normalizedDraft.llm,provider:normalizedDraft.llm.provider});
    setConstraints(nextConstraints);
    setDemographic(normalizedDraft.demographic);
    setDemographicPrompts({...normalizedDraft.demographicPrompts});
    setMaxLoras(cappedMax);
    setWidth(normalizedDraft.width);
    setHeight(normalizedDraft.height);
    setSteps(normalizedDraft.steps);
    setCfg(normalizedDraft.cfg);
    setSampler(normalizedDraft.sampler);
    setMinPositiveTags(normalizedDraft.minPositiveTags);
    setMinNegativeTags(normalizedDraft.minNegativeTags);
    try{
      window.localStorage.setItem(
        generationSettingsStorageKey,
        JSON.stringify({...normalizedDraft,manualLoraIds}),
      );
      window.localStorage.setItem(demographicPromptsStorageKey,JSON.stringify(normalizedDraft.demographicPrompts));
    }catch{}
  },[generationDraft]);

  useEffect(()=>{
    if(isTauriRuntime){
      if(!webHost) return;
    }else if(!lanClientReady){
      return;
    }

    const timeout=window.setTimeout(()=>{
      void apiInvoke('update_web_host_llm',{llmSettings:generationDraft.llm})
        .catch(e=>setWebHostError(String(e)));
    },400);
    return ()=>window.clearTimeout(timeout);
  },[generationDraft.llm,webHost,lanClientReady]);

  useEffect(()=>{
    // The desktop host publishes its settings while hosting; LAN clients also
    // write intentional settings changes back to that same host-side store.
    if(isTauriRuntime ? !webHost : !lanClientReady) return;
    generationSettingsEditedAt.current=Date.now();
    const timeout=window.setTimeout(()=>{
      void apiInvoke('update_web_host_generation_settings',{generationSettings:generationDraft})
        .catch(e=>setWebHostError('Could not sync generation settings with the host: '+String(e)));
    },450);
    return ()=>window.clearTimeout(timeout);
  },[generationDraft,webHost,lanClientReady]);

  useEffect(()=>{
    // Keep both the LAN client and the desktop host aligned with the host-side
    // settings store. This also persists remote edits through the host's normal
    // localStorage effect so they survive a desktop restart.
    if(isTauriRuntime ? !webHost : !lanClientReady) return;
    let active=true;
    const refresh=async()=>{
      try{
        const hostConfig=await apiInvoke<{
          provider:ProviderKind;
          baseUrl:string;
          model:string;
          temperature?:number;
          maxTokens?:number;
          contextTokens?:number;
          generationSettings?:Partial<StoredGenerationSettings>;
        }>(isTauriRuntime ? 'get_web_host_shared_settings' : 'get_host_llm_config');
        if(!active || Date.now()-generationSettingsEditedAt.current<2000) return;
        const hostGeneration=hostConfig.generationSettings || {};
        const hostLlm={
          provider:hostConfig.provider,
          baseUrl:hostConfig.baseUrl,
          model:hostConfig.model || '',
          temperature:hostConfig.temperature,
          maxTokens:hostConfig.maxTokens,
          contextTokens:hostConfig.contextTokens,
        };
        setProvider(hostConfig.provider);
        setLlm(current=>{
          const next={
            ...current,
            provider:hostLlm.provider,
            baseUrl:hostLlm.baseUrl,
            model:hostLlm.model,
            temperature:hostLlm.temperature ?? current.temperature,
            maxTokens:hostLlm.maxTokens ?? current.maxTokens,
            contextTokens:hostLlm.contextTokens ?? current.contextTokens,
          };
          return current.provider===next.provider
            && current.baseUrl===next.baseUrl
            && current.model===next.model
            && current.temperature===next.temperature
            && current.maxTokens===next.maxTokens
            && current.contextTokens===next.contextTokens
            ? current : next;
        });
        setGenerationDraft(current=>{
          const currentSignature=JSON.stringify({
            plannerSystemPrompt:current.plannerSystemPrompt,
            tagSystemPrompt:current.tagSystemPrompt,
            repairSystemPrompt:current.repairSystemPrompt,
            userPromptTemplate:current.userPromptTemplate,
            expansionPromptTemplate:current.expansionPromptTemplate,
            demographic:current.demographic,
            demographicPrompts:current.demographicPrompts,
            minPositiveTags:current.minPositiveTags,
            minNegativeTags:current.minNegativeTags,
            maxNegativeTags:current.maxNegativeTags,
            maxTagLength:current.maxTagLength,
            plannerTemperature:current.plannerTemperature,
            tagTemperature:current.tagTemperature,
            tagGenerationRetries:current.tagGenerationRetries,
            maxCharacterLoras:current.maxCharacterLoras,
            maxLoras:current.maxLoras,
            randomLoraMin:current.randomLoraMin,
            randomLoraMax:current.randomLoraMax,
            constraints:current.constraints,
            width:current.width,
            height:current.height,
            steps:current.steps,
            cfg:current.cfg,
            sampler:current.sampler,
          });
          const hostSignature=JSON.stringify({
            plannerSystemPrompt:hostGeneration.plannerSystemPrompt,
            tagSystemPrompt:hostGeneration.tagSystemPrompt,
            repairSystemPrompt:hostGeneration.repairSystemPrompt,
            userPromptTemplate:hostGeneration.userPromptTemplate,
            expansionPromptTemplate:hostGeneration.expansionPromptTemplate,
            demographic:hostGeneration.demographic,
            demographicPrompts:hostGeneration.demographicPrompts,
            minPositiveTags:hostGeneration.minPositiveTags,
            minNegativeTags:hostGeneration.minNegativeTags,
            maxNegativeTags:hostGeneration.maxNegativeTags,
            maxTagLength:hostGeneration.maxTagLength,
            plannerTemperature:hostGeneration.plannerTemperature,
            tagTemperature:hostGeneration.tagTemperature,
            tagGenerationRetries:hostGeneration.tagGenerationRetries,
            maxCharacterLoras:hostGeneration.maxCharacterLoras,
            maxLoras:hostGeneration.maxLoras,
            randomLoraMin:hostGeneration.randomLoraMin,
            randomLoraMax:hostGeneration.randomLoraMax,
            constraints:hostGeneration.constraints,
            width:hostGeneration.width,
            height:hostGeneration.height,
            steps:hostGeneration.steps,
            cfg:hostGeneration.cfg,
            sampler:hostGeneration.sampler,
          });
          const currentLlm={
            provider:current.llm.provider,
            baseUrl:current.llm.baseUrl,
            model:current.llm.model,
            temperature:current.llm.temperature,
            maxTokens:current.llm.maxTokens,
            contextTokens:current.llm.contextTokens,
          };
          const targetLlm={
            provider:hostLlm.provider,
            baseUrl:hostLlm.baseUrl,
            model:hostLlm.model,
            temperature:hostLlm.temperature ?? current.llm.temperature,
            maxTokens:hostLlm.maxTokens ?? current.llm.maxTokens,
            contextTokens:hostLlm.contextTokens ?? current.llm.contextTokens,
          };
          const generationChanged=currentSignature!==hostSignature;
          const llmChanged=JSON.stringify(currentLlm)!==JSON.stringify(targetLlm);
          if(!generationChanged && !llmChanged) return current;
          return {
            ...current,
            ...(generationChanged ? hostGeneration : {}),
            llm:llmChanged ? {...current.llm,...targetLlm} : current.llm,
          };
        });
      }catch(e){
        if(active) setWebHostError('Could not refresh shared host settings: '+String(e));
      }
    };
    void refresh();
    const interval=window.setInterval(()=>void refresh(),2500);
    return ()=>{
      active=false;
      window.clearInterval(interval);
    };
  },[webHost,lanClientReady]);

  useEffect(()=>{
    // LAN generations append to the host's persistent history file. Refresh the
    // desktop view while hosting so remote generations appear without a restart.
    // On LAN clients, refresh while the History tab is open to show host records.
    const shouldRefreshHistory=isTauriRuntime
      ? Boolean(webHost) || tab==='history'
      : tab==='history';
    if(!shouldRefreshHistory) return;
    void loadHistory(tab==='history');
    const interval=window.setInterval(()=>void loadHistory(),2500);
    return ()=>window.clearInterval(interval);
  },[tab,webHost]);


  async function rollStack():Promise<string[]|null>{
    if(!selected || !library){
      setError('Select a checkpoint first.');
      return null;
    }
    setError('');

    const manualIds=manualLoraIds.filter(id=>allLoras.some(lora=>lora.id===id));
    const manualCharacterCount=manualIds.reduce((count,id)=>{
      const lora=allLoras.find(item=>item.id===id);
      return count+(lora && isCharacterLoraForCheckpoint(lora,selected) ? 1 : 0);
    },0);
    if(manualCharacterCount>generationDraft.maxCharacterLoras){
      setError('The selected manual stack contains '+manualCharacterCount+' character LoRAs, but the configured maximum is '+generationDraft.maxCharacterLoras+'.');
      return null;
    }
    const manualHasCharacter=manualCharacterCount>0;

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
    setPlan(null);
    setPlanStream('');
    setTagValidation('');
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
      const availableCharacterSlots=Math.max(0,generationDraft.maxCharacterLoras-manualCharacterCount);

      if(!manualHasCharacter && randomSlots>0 && characterPool.length===0){
        lastError=new Error('No compatible character LoRA was found for the selected base-model tag in the Registry.');
        break;
      }

      const characterSlots=Math.min(
        availableCharacterSlots,
        randomSlots,
        characterPool.length,
      );
      const randomCharacterCount=characterSlots>0
        ? Math.floor(Math.random()*characterSlots)+1
        : 0;

      randomIds.push(...characterPool.slice(0,randomCharacterCount).map(lora=>lora.id));

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
            maxCharacterLoras:generationDraft.maxCharacterLoras,
            registryUrl:registryUrl || null,
          },
        });
        setSelectedLoraIds(combinedIds);
        setPrepared(result);
        setStageStatus('compatibility','done');
        setStage('selection');
        setStageStatus('selection','done');
        return combinedIds;
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
      setError('');
    }else{
      const lora=allLoras.find(item=>item.id===id);
      if(!lora) return;
      if(selectedLoraIds.length>=generationDraft.maxLoras){
        setError('Maximum LoRAs reached. Increase MAX LoRAs in Settings or remove an existing LoRA.');
        return;
      }
      const isCharacter=lora.character || lora.tags.some(tag=>normUi(tag)==='character');
      if(isCharacter && selectedCharacterLoraCount>=generationDraft.maxCharacterLoras){
        setError('Maximum character LoRAs reached. Increase MAX CHARACTER LoRAs in Settings or remove an existing character LoRA.');
        return;
      }
      setSelectedLoraIds(current=>[...current,id]);
      setManualLoraIds(current=>[...current,id]);
      setError('');
    }
    setPrepared(null);
    setPrompts(null);
  }

  const [autoGenerating,setAutoGenerating]=useState(false);
  const autoGeneratingRef=useRef(false);

  async function stopGeneration(){
    autoGeneratingRef.current=false;
    setAutoGenerating(false);
    if(!busy) {
      setToast('NO ACTIVE GENERATION');
      return;
    }
    setToast('STOPPING GENERATION…');
    try{
      await apiInvoke('stop_generation',{comfyUrl});
    }catch(e){
      setError('Could not stop generation: '+String(e));
    }
  }

  function stopAutoGenerate(){
    autoGeneratingRef.current=false;
    setAutoGenerating(false);
    if(busy){
      void stopGeneration();
    }else{
      setToast('AUTO GENERATE STOPPED');
    }
  }

  async function autoGenerate(){
    if(autoGeneratingRef.current || busy || !selected || !library || !llm.model) return;
    autoGeneratingRef.current=true;
    setAutoGenerating(true);
    setError('');
    setToast('AUTO GENERATE STARTED');

    let stackNumber=0;
    try{
      while(autoGeneratingRef.current){
        stackNumber+=1;
        const stackIds=await rollStack();
        if(!stackIds || !autoGeneratingRef.current) break;

        let attempt=0;
        let success=false;
        while(autoGeneratingRef.current && !success){
          attempt+=1;
          setToast('AUTO STACK '+stackNumber+' · ATTEMPT '+attempt+' · GENERATING');
          success=await generate(stackIds);
          if(!success && autoGeneratingRef.current){
            setToast('AUTO STACK '+stackNumber+' · ATTEMPT '+attempt+' FAILED · RETRYING SAME LORA STACK');
            await new Promise(resolve=>setTimeout(resolve,250));
          }
        }

        if(success && autoGeneratingRef.current){
          setToast('AUTO STACK '+stackNumber+' COMPLETE · SELECTING NEXT RANDOM LORA STACK');
        }
      }
    }catch(e){
      setError(String(e));
    }finally{
      autoGeneratingRef.current=false;
      setAutoGenerating(false);
    }
  }

  async function generate(loraIdsOverride?:string[]):Promise<boolean>{
    if(busy || !selected || !library) return false;
    setBusy(true);
    setError('');
    try{
      await apiInvoke('start_generation');
    }catch(e){
      setBusy(false);
      setError('Could not start generation: '+String(e));
      return false;
    }
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
    let generationSucceeded=false;

    try{
      const sourceLoraIds=loraIdsOverride ?? selectedLoraIds;
      const generationSelectedLoraIds=sourceLoraIds.filter(id=>compatibleLoras.some(lora=>lora.id===id));

      setStage('compatibility');
      setStageStatus('compatibility','running');
      const prep=await apiInvoke<PreparedGeneration>('prepare_generation',{
        req:{
          checkpoint:selected,
          loras:library.loras,
          selectedLoraIds:generationSelectedLoraIds,
          ...constraints,
          randomLoraMax:Math.min(maxLoras,constraints.randomLoraMax),
          maxCharacterLoras:generationDraft.maxCharacterLoras,
          registryUrl:registryUrl || null,
        },
      });
      setPrepared(prep);
      setStageStatus('compatibility','done');
      setStage('selection');
      setStageStatus('selection','done');
      setPlan(null);
      setPlanStream('');
      setTagValidation('');

      const loraMetadata=prep.loras.map((l,index)=>
        'LORA ' + (index+1) + '\n' +
        'NAME: ' + l.name + '\n' +
        'TYPE: ' + (l.character ? 'CHARACTER IDENTITY' : 'SUPPORTING CONCEPT') + '\n' +
        'BASE MODEL: ' + (l.baseModel || 'unknown') + '\n' +
        'TAGS: ' + (l.tags.length ? l.tags.join(', ') : '(none)') + '\n' +
        'DESCRIPTION: ' + (l.description || '(none)') + '\n' +
        'ACTIVATION PROMPT(S): ' + (l.activationTags.length ? l.activationTags.join(' | ') : '(none)')
      ).join('\n\n');

      // Each LLM stage has its own editable system prompt.
      // The selected demographic policy is inserted into the stage prompt.
      const selectedDemographicPolicy = demographicPrompts[demographic];
      const systemPromptValues={
        DEMOGRAPHIC_POLICY:selectedDemographicPolicy,
        MIN_POSITIVE_TAGS:String(generationDraft.minPositiveTags),
        MIN_NEGATIVE_TAGS:String(generationDraft.minNegativeTags),
        MAX_NEGATIVE_TAGS:String(generationDraft.maxNegativeTags),
        MAX_TAG_LENGTH:String(generationDraft.maxTagLength),
      };
      const plannerSystemPrompt = renderPromptTemplate(
        generationDraft.plannerSystemPrompt,
        systemPromptValues,
      );
      const tagSystemPrompt = renderPromptTemplate(
        generationDraft.tagSystemPrompt,
        systemPromptValues,
      );
      const repairSystemPrompt = renderPromptTemplate(
        generationDraft.repairSystemPrompt,
        systemPromptValues,
      );

      const characterLoras=prep.loras.filter(l=>
        l.character || l.tags.some(tag=>normUi(tag)==='character')
      );
      const selectedCharacterInputs:SelectedCharacterInput[]=characterLoras.map(l=>({
        name:l.name,
        description:l.description,
        activationTags:l.activationTags,
      }));
      const characterMetadata=characterLoras.length
        ? characterLoras.map((l,index)=>
            'CHARACTER '+(index+1)+' LoRA\n'+
            'NAME: '+l.name+'\n'+
            'DESCRIPTION: '+(l.description || '(none)')+'\n'+
            'ACTIVATION PROMPT(S): '+(l.activationTags.length ? l.activationTags.join(' | ') : '(none)')
          ).join('\n\n')
        : '(none selected)';

      const plannerAnchorType=pickPlannerAnchorType(constraints);
      const basePromptValues={
        CHECKPOINT:prep.checkpoint.name,
        BASE:prep.checkpoint.baseModel || 'unknown',
        SCENE_ANCHOR_TYPE:plannerAnchorType,
        COMPATIBILITY:prep.compatibilityKeys.join(', '),
        LORA_METADATA:loraMetadata,
        CHARACTERS:characterMetadata,
        CHARACTER:characterMetadata,
        CHARACTER_COUNT:String(characterLoras.length),
        SETTING:prep.scene.setting,
        POSE:prep.scene.pose,
        EXPRESSION:prep.scene.expression,
        DRESS:prep.scene.dress,
        COMPOSITION:prep.scene.composition,
        EXTRA:constraints.additional || '(none)',
      };

      setStage('planning');
      setStageStatus('planning','running');
      streamText.current='';
      setPlanStream('');
      await streamLlm({
        settings:{...llm,temperature:generationDraft.plannerTemperature},
        systemPrompt:plannerSystemPrompt,
        userPrompt:renderPromptTemplate(defaultPlanningPrompt,basePromptValues),
      },event=>{
        streamText.current+=event;
        flushSync(()=>setPlanStream(streamText.current));
      });

      const generationPlan=parseGenerationPlan(streamText.current,prep.scene,selectedCharacterInputs,plannerAnchorType);
      setPlan(generationPlan);
      setStageStatus('planning','done');

      const planValues={
        ...basePromptValues,
        CHARACTER:generationPlan.characters.map((character,index)=>
          'CHARACTER '+(index+1)+': '+character.name+
          ' | GENDER: '+character.gender+
          ' | APPEARANCE: '+character.appearance+
          ' | POSE: '+character.pose+
          ' | EXPRESSION: '+character.expression+
          ' | POSITION: '+character.position+
          ' | INTERACTION: '+character.interaction
        ).join('\n'),
        CHARACTERS:generationPlan.characters.map((character,index)=>
          'CHARACTER '+(index+1)+': '+character.name+
          ' | GENDER: '+character.gender+
          ' | APPEARANCE: '+character.appearance+
          ' | POSE: '+character.pose+
          ' | EXPRESSION: '+character.expression+
          ' | POSITION: '+character.position+
          ' | INTERACTION: '+character.interaction
        ).join('\n'),
        SETTING:generationPlan.setting,
        POSE:generationPlan.pose,
        EXPRESSION:generationPlan.expression,
        DRESS:generationPlan.dress,
        COMPOSITION:generationPlan.composition,
        ACTION:generationPlan.action,
        BACKGROUND:generationPlan.background,
        LIGHTING:generationPlan.lighting,
        CAMERA:generationPlan.camera,
        FRAMING:generationPlan.framing,
      };

      setStage('llm');
      setStageStatus('llm','running');
      streamText.current='';
      setStream('');
      setPrompts(null);
      const plannedDecisionBlock=[
        'PLANNED GENERATION DECISION:',
        'SCENE ANCHOR ['+generationPlan.anchorType.toUpperCase()+']: '+generationPlan.anchor,
        'CENTRAL CONCEPT: '+generationPlan.concept,
        'CHARACTERS:',
        ...generationPlan.characters.map((character,index)=>
          'CHARACTER '+(index+1)+': '+character.name+
          ' | GENDER: '+character.gender+
          ' | APPEARANCE: '+character.appearance+
          ' | POSE: '+character.pose+
          ' | EXPRESSION: '+character.expression+
          ' | POSITION: '+character.position+
          ' | INTERACTION: '+character.interaction
        ),
        'CONCEPT: '+generationPlan.concept,
        'ACTION: '+generationPlan.action,
        'POSE: '+generationPlan.pose,
        'SETTING: '+generationPlan.setting,
        'BACKGROUND: '+generationPlan.background,
        'EXPRESSION: '+generationPlan.expression,
        'DRESS: '+generationPlan.dress,
        'COMPOSITION: '+generationPlan.composition,
        'LIGHTING: '+generationPlan.lighting,
        'CAMERA: '+generationPlan.camera,
        'FRAMING: '+generationPlan.framing,
      ].join('\n');

      const userPrompt=renderPromptTemplate(
        generationDraft.userPromptTemplate,
        planValues,
      )+'\n\n'+plannedDecisionBlock+
        '\n\nConvert this exact plan into deterministic comma-separated image tags. Do not invent a different pose, background, expression, lighting, camera or framing.';

      await streamLlm({
        settings:{...llm,temperature:generationDraft.tagTemperature},
        systemPrompt:tagSystemPrompt,
        userPrompt,
      },event=>{
        streamText.current+=event;
        flushSync(()=>setStream(streamText.current));
      });

      let rawPair=await apiInvoke<PromptPair>('parse_prompt_pair',{raw:streamText.current});
      let validation=validateGeneratedPrompt(
        rawPair,
        prep.loras.flatMap(l=>l.activationTags),
        generationDraft.minPositiveTags,
        generationDraft.minNegativeTags,
        generationDraft.maxNegativeTags,
        generationDraft.maxTagLength,
      );
      let validationAttempt=0;

      while((!validation.valid || promptNeedsExpansion(rawPair)) && validationAttempt<generationDraft.tagGenerationRetries){
        validationAttempt+=1;
        const deficiency=[
          ...validation.errors,
          ...(promptNeedsExpansion(rawPair) ? ['prompt does not contain enough scene coverage or visual detail'] : []),
        ];
        setTagValidation('TAG VALIDATION FAILED — REDO '+validationAttempt+'/'+generationDraft.tagGenerationRetries+': '+deficiency.join(' · '));
        streamText.current='';
        flushSync(()=>setStream(''));

        const expansionUserPrompt=renderPromptTemplate(
          generationDraft.expansionPromptTemplate,
          {
            ...planValues,
            PREVIOUS_JSON:JSON.stringify(rawPair),
          },
        )+
        '\n\nSTRICT VALIDATION FAILURE. REDO THE OUTPUT NOW.\n'+
        deficiency.map(item=>'- '+item).join('\n')+
        '\nRules: at least '+generationDraft.minPositiveTags+' positive comma-separated tags, between '+generationDraft.minNegativeTags+' and '+generationDraft.maxNegativeTags+' negative tags, ordinary tags must be 1-6 words and <='+generationDraft.maxTagLength+' characters, no sentence-like tags, no prose, no metaphors, no narrative clauses. Preserve every documented activation prompt exactly. Return JSON only.';

        await streamLlm({
          settings:{...llm,temperature:generationDraft.tagTemperature},
          systemPrompt:repairSystemPrompt,
          userPrompt:expansionUserPrompt,
        },event=>{
          streamText.current+=event;
          flushSync(()=>setStream(streamText.current));
        });

        rawPair=await apiInvoke<PromptPair>('parse_prompt_pair',{raw:streamText.current});
        validation=validateGeneratedPrompt(
          rawPair,
          prep.loras.flatMap(l=>l.activationTags),
          generationDraft.minPositiveTags,
          generationDraft.minNegativeTags,
          generationDraft.maxNegativeTags,
          generationDraft.maxTagLength,
        );
      }

      if(!validation.valid || promptNeedsExpansion(rawPair)){
        throw new Error(
          'LLM prompt validation failed after '+generationDraft.tagGenerationRetries+' repair attempts: '+
          [...validation.errors, ...(promptNeedsExpansion(rawPair) ? ['insufficient scene coverage'] : [])].join(' · ')
        );
      }

      setTagValidation('TAG VALIDATION PASSED — '+validation.positiveTags.length+' positive / '+validation.negativeTags.length+' negative tags');

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
          generationSucceeded=true;
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
        systemPrompt:tagSystemPrompt,
        plannerSystemPrompt,
        tagSystemPrompt,
        repairSystemPrompt,
        demographic,
        demographicPrompts:{...demographicPrompts},
        userPromptTemplate:generationDraft.userPromptTemplate,
        expansionPromptTemplate:generationDraft.expansionPromptTemplate,
        maxLoras,
        randomLoraMin:constraints.randomLoraMin,
        randomLoraMax:constraints.randomLoraMax,
        constraints:{...constraints},
        width,
        height,
        steps,
        cfg,
        sampler,
        minPositiveTags:generationDraft.minPositiveTags,
        minNegativeTags:generationDraft.minNegativeTags,
        maxNegativeTags:generationDraft.maxNegativeTags,
        maxTagLength:generationDraft.maxTagLength,
        plannerTemperature:generationDraft.plannerTemperature,
        tagTemperature:generationDraft.tagTemperature,
        tagGenerationRetries:generationDraft.tagGenerationRetries,
        maxCharacterLoras:generationDraft.maxCharacterLoras,
      };
      const record:GenerationRecord={
        id:crypto.randomUUID(),
        timestamp:new Date().toISOString(),
        provider:llm.provider,
        model:llm.model,
        checkpoint:prep.checkpoint,
        loras:prep.loras,
        scene:plan ? {
          character:plan.character,
          setting:plan.setting,
          pose:plan.pose,
          expression:plan.expression,
          dress:plan.dress,
          composition:plan.composition,
        } : prep.scene,
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
      setToast(generationSucceeded
        ? (promptId ? 'Generation complete · ' + promptId : 'Generation complete')
        : (promptId ? 'Generation recorded without an output image · ' + promptId : 'Generation failed without an output image'));
      return generationSucceeded;
    }catch(e){
      setError(String(e));
      setStageStatus(stage,'error');
      return generationSucceeded;
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
    setMobileControlsOpen(false);
    setSelectedId(id);
    setPrepared(null);
    setPrompts(null);
    setPlan(null);
    setPlanStream('');
    setTagValidation('');
  }

  async function copy(text:string){
    try{
      if(navigator.clipboard?.writeText){
        await navigator.clipboard.writeText(text);
      }else{
        const helper=document.createElement('textarea');
        helper.value=text;
        helper.style.position='fixed';
        helper.style.opacity='0';
        document.body.appendChild(helper);
        helper.focus();
        helper.select();
        document.execCommand('copy');
        helper.remove();
      }
      setToast('Copied');
    }catch{
      setToast('Copy unavailable — long-press the address to copy it.');
    }
  }

  async function startLanHost(){
    if(!isTauriRuntime) return;
    setWebHostBusy(true);
    setWebHostError('');
    try{
      const info=await apiInvoke<WebHostInfo>('start_web_host',{
        port:1424,
        llmSettings:llm,
        generationSettings:generationDraft,
      });
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
        <div className="mobile-actions">
          <button type="button" className={mobileControlsOpen ? 'active' : ''} onClick={()=>setMobileControlsOpen(v=>!v)} title="Open model and backend controls">
            <Layers3 size={13}/> MODELS
          </button>
          <button type="button" className={tab==='history' ? 'active' : ''} onClick={()=>{setTab('history');setSettingsOpen(false);setMobileControlsOpen(false)}} title="Open generation history">
            <History size={13}/> HISTORY
          </button>
          <button type="button" className={settingsOpen ? 'active' : ''} onClick={()=>{setSettingsOpen(v=>!v);setMobileControlsOpen(false)}} title="Open settings">
            <Settings2 size={13}/> SETTINGS
          </button>
        </div>
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

      <div className="showcase-heading">
        <span>IMAGE SHOWCASE</span>
        <span>{history.filter(item=>item.imageDataUrl).length}</span>
      </div>
      <div className="image-showcase">
        <div className="showcase-main">
          {(resultImage || history.find(item=>item.imageDataUrl)?.imageDataUrl)
            ? <img src={resultImage || history.find(item=>item.imageDataUrl)?.imageDataUrl || ''} alt="Latest generated result"/>
            : <div className="showcase-empty"><WandSparkles size={22}/><span>NO GENERATED IMAGE</span></div>}
        </div>
        <div className="showcase-meta">
          <b>{resultFilename || history.find(item=>item.imageDataUrl)?.checkpoint.name || 'READY FOR GENERATION'}</b>
          <span>{comfyStatus==='done' ? 'LATEST GENERATION' : selected?.name || 'SELECT A CHECKPOINT'}</span>
        </div>
        <div className="showcase-strip">
          {history.filter(item=>item.imageDataUrl).slice(0,6).map(item=>
            <button className="showcase-thumb" key={item.id} onClick={()=>openHistory(item.id)} title={new Date(item.timestamp).toLocaleString()}>
              <img src={item.imageDataUrl} alt=""/>
            </button>
          )}
          {!history.some(item=>item.imageDataUrl) && <div className="showcase-no-history">NO HISTORY</div>}
        </div>
      </div>

      {webHost && <div className="host-box">
        <div className="root-label">WEB HOST</div>
        <div className="host-url">{webHost.lanUrl}</div>
        <button onClick={()=>void copy(webHost.lanUrl)}><Copy size={12}/> COPY ADDRESS</button>
      </div>}


    </aside>

    {mobileControlsOpen && <div className="mobile-controls-backdrop" onMouseDown={()=>setMobileControlsOpen(false)}/>}
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

          <div className="planning-panel">
            <div className="planning-head">
              <span>LLM VISUAL PLAN</span>
              {plan && <strong>LOCKED</strong>}
            </div>
            <div className="planning-stream-box">
              {planStream ? <pre>{planStream}</pre> : <div className="stream-placeholder"><Terminal size={18}/><span>Planning decision will stream here first.</span></div>}
            </div>
            {plan && <div className="planning-grid">
              <div>
                <span>SCENE ANCHOR · {plan.anchorType.toUpperCase()}</span>
                <b>{plan.anchor}</b>
              </div>
              <div>
                <span>CENTRAL CONCEPT</span>
                <b>{plan.concept}</b>
              </div>
              <div className="planning-character-list">
                <span>CHARACTERS · {selectedCharacterLoras.length}</span>
                {selectedCharacterLoras.map((lora,index)=>{
                  const character=plan.characters[index];
                  return (
                    <div key={lora.id} className="planning-character-card">
                      <b>{index+1}. {lora.name}</b>
                      <small>GENDER: {character?.gender || 'unknown'}</small>
                      <small>{character?.appearance || lora.cacheDescription || lora.name || 'character identity LoRA selected'}</small>
                      <small>POSE: {character?.pose || plan.pose} · EXPRESSION: {character?.expression || plan.expression}</small>
                      <small>POSITION: {character?.position || 'planned position'} · {character?.interaction || 'planned interaction'}</small>
                    </div>
                  );
                })}
              </div>
              <div><span>ACTION</span><b>{plan.action}</b></div>
              <div><span>POSE</span><b>{plan.pose}</b></div>
              <div><span>BACKGROUND</span><b>{plan.background}</b></div>
              <div><span>EXPRESSION</span><b>{plan.expression}</b></div>
              <div><span>DRESS</span><b>{plan.dress}</b></div>
              <div><span>COMPOSITION</span><b>{plan.composition}</b></div>
              <div><span>LIGHTING</span><b>{plan.lighting}</b></div>
              <div><span>CAMERA</span><b>{plan.camera}</b></div>
              <div><span>FRAMING</span><b>{plan.framing}</b></div>
            </div>}
          </div>

          <div className="stream-box tag-stream-box">{stream
            ? <pre>{stream}</pre>
            : <div className="stream-placeholder"><Terminal size={18}/><span>Deterministic image tags appear here after planning.</span></div>}
          </div>
          {tagValidation && <div className="tag-validation">{tagValidation}</div>}

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
            <button className="secondary-btn" disabled={busy || autoGenerating || !selected} onClick={()=>void rollStack()}><RefreshCw size={14}/> RANDOMIZE LORAS</button>
            {busy
              ? <button className="secondary-btn active" onClick={()=>void stopGeneration()}><X size={15}/> STOP GENERATION</button>
              : <button className="primary-btn" disabled={autoGenerating || !selected || !llm.model} onClick={()=>void generate()}><Play size={15}/> GENERATE</button>}
            <button className={'secondary-btn ' + (autoGenerating ? 'active' : '')} disabled={!selected || !llm.model} onClick={()=>autoGenerating ? stopAutoGenerate() : void autoGenerate()}>
              <Sparkles size={14}/> {autoGenerating ? 'STOP AUTO' : 'AUTO GENERATE'}
            </button>
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
        <div className="history-mobile-nav">
          <button className="ghost-btn" onClick={()=>{setTab('generate');setSelectedHistoryId('')}}><WandSparkles size={13}/> GENERATE</button>
          <button className="ghost-btn" onClick={()=>void loadHistory(true)}><RefreshCw size={13}/> REFRESH HISTORY</button>
        </div>
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
              <div><span>PLANNER TEMPERATURE</span><b>{(selectedHistory.generationSettings.plannerTemperature ?? selectedHistory.generationSettings.llm.temperature).toFixed(2)}</b></div>
              <div><span>TAG GENERATOR TEMPERATURE</span><b>{(selectedHistory.generationSettings.tagTemperature ?? selectedHistory.generationSettings.llm.temperature).toFixed(2)}</b></div>
              <div><span>TAG GENERATION RETRIES</span><b>{selectedHistory.generationSettings.tagGenerationRetries ?? defaultTagGenerationRetries}</b></div>
              <div><span>BASE TEMPERATURE</span><b>{selectedHistory.generationSettings.llm.temperature.toFixed(2)}</b></div>
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

            <div className="history-section-title">PLANNER SYSTEM PROMPT · SENT EXACTLY TO MODEL</div>
            <pre className="history-code">{selectedHistory.generationSettings.plannerSystemPrompt || selectedHistory.generationSettings.systemPrompt}</pre>
            <div className="history-section-title">TAG GENERATION SYSTEM PROMPT · SENT EXACTLY TO MODEL</div>
            <pre className="history-code">{selectedHistory.generationSettings.tagSystemPrompt || selectedHistory.generationSettings.systemPrompt}</pre>
            <div className="history-section-title">TAG REPAIR SYSTEM PROMPT · SENT EXACTLY TO MODEL</div>
            <pre className="history-code">{selectedHistory.generationSettings.repairSystemPrompt || selectedHistory.generationSettings.systemPrompt}</pre>
            <div className="history-section-title">PRIMARY USER PROMPT TEMPLATE</div>
            <pre className="history-code">{selectedHistory.generationSettings.userPromptTemplate || defaultUserPromptTemplate}</pre>
            <div className="history-section-title">EXPANSION USER PROMPT TEMPLATE</div>
            <pre className="history-code">{selectedHistory.generationSettings.expansionPromptTemplate || defaultExpansionPromptTemplate}</pre>
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

    <aside className={'right-rail ' + (mobileControlsOpen ? 'mobile-open' : '')}>
      <button type="button" className="mobile-controls-close" onClick={()=>setMobileControlsOpen(false)}><X size={14}/> CLOSE CONTROLS</button>
      <div className="rail-label">MODELS</div>
      {selected && <div className="selected-model-card">
        <div className="selected-model-thumb"><ModelThumbnail model={selected} iconSize={22}/></div>
        <div className="selected-model-copy"><b>{selected.name}</b><span>{selected.baseModel || 'BASE UNKNOWN'}</span></div>
      </div>}

      <div className="right-section">
        <div className="right-section-head"><span>CHECKPOINTS</span><span>{filteredCheckpoints.length}</span></div>
        <div className="search-row compact"><Search size={13}/><input value={checkpointSearch} onChange={e=>setCheckpointSearch(e.target.value)} placeholder="Search checkpoints"/></div>
        <div className="checkpoint-list">
          {filteredCheckpoints.map(model=>{
            const duplicate=duplicateCheckpointNames.get(model.name.trim().toLowerCase()) || 0;
            const filename=model.path.split(/[\\/]/).pop() || model.path;
            return <button key={model.id} title={[model.name,model.baseModel || '',filename,...model.tags].filter(Boolean).join(' · ')} className={'checkpoint-row ' + (selected?.id===model.id ? 'selected' : '')} onClick={()=>selectCheckpoint(model.id)}>
              <div className="checkpoint-row-thumb"><ModelThumbnail model={model} iconSize={16}/></div>
              <div className="checkpoint-row-copy"><b>{model.name}</b><span>{model.baseModel || 'BASE UNKNOWN'}</span><small>{duplicate>1 ? filename : model.tags.slice(0,3).join(' · ')}</small></div>
              {selected?.id===model.id && <Check size={14}/>}
            </button>;
          })}
        </div>
      </div>

      <div className="right-section lora-section">
        <div className="right-section-head"><span>LORAS</span><span>{selectedLoraIds.length} / {generationDraft.maxLoras} · {selectedCharacterLoraCount} CHAR</span></div>
        <div className="search-row compact"><Search size={13}/><input value={loraSearch} onChange={e=>setLoraSearch(e.target.value)} placeholder="Search LoRAs"/></div>
        <div className="lora-list">
          {filteredLoras.map(lora=>{
            const duplicate=duplicateLoraNames.get(lora.name.trim().toLowerCase()) || 0;
            const filename=lora.path.split(/[\\/]/).pop() || lora.path;
            return <button key={lora.id} title={[lora.name,lora.baseModel || '',filename,...lora.tags].filter(Boolean).join(' · ')} className={'lora-row ' + (selectedLoraIds.includes(lora.id) ? 'selected' : '')} onClick={()=>toggleLora(lora.id)}>
              <div className="lora-row-thumb"><ModelThumbnail model={lora} iconSize={15}/></div>
              <div className="lora-row-copy"><b>{lora.name}</b><span>{lora.character ? 'CHARACTER' : 'SUPPORT'}</span><small>{duplicate>1 ? filename : lora.tags.slice(0,3).join(' · ')}</small></div>
              {selectedLoraIds.includes(lora.id) && <Check size={14}/>}
            </button>;
          })}
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

        <div className="settings-tabs" role="tablist" aria-label="Generation settings">
          {([
            ['llm','LLM'],
            ['system-prompts','SYSTEM PROMPTS'],
            ['prompt-templates','PROMPT TEMPLATES'],
            ['scene','SCENE'],
            ['output','OUTPUT'],
          ] as const).map(([key,label])=>
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={settingsTab===key}
              className={'settings-tab ' + (settingsTab===key ? 'active' : '')}
              onClick={()=>setSettingsTab(key)}
            >
              {label}
            </button>
          )}
        </div>

        <div className="drawer-scroll">
          {settingsTab==='llm' && <div className="settings-tab-panel">
            <section className="settings-section settings-card">
              <div className="settings-section-title">MODEL PROVIDER</div>
              <div className="provider-toggle">
                <button type="button" className={generationDraft.llm.provider==='ollama' ? 'active' : ''} onClick={()=>setGenerationDraft(d=>({...d,llm:{...d.llm,provider:'ollama',baseUrl:'http://127.0.0.1:11434'}}))}>OLLAMA</button>
                <button type="button" className={generationDraft.llm.provider==='openai-compatible' ? 'active' : ''} onClick={()=>setGenerationDraft(d=>({...d,llm:{...d.llm,provider:'openai-compatible',baseUrl:'http://127.0.0.1:8080/v1'}}))}>OPENAI COMPATIBLE</button>
              </div>
              <div className="settings-form-grid">
                <label className="settings-field settings-field-full"><span>BASE URL</span><input value={generationDraft.llm.baseUrl} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,baseUrl:e.target.value}}))}/></label>
                <label className="settings-field settings-field-full"><span>API KEY</span><input type="password" value={generationDraft.llm.apiKey} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,apiKey:e.target.value}}))}/></label>
                <label className="settings-field settings-field-full"><span>MODEL</span>
                  <select value={generationDraft.llm.model} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,model:e.target.value}}))}>
                    <option value="">SELECT MODEL</option>{models.map(model=><option key={model}>{model}</option>)}
                  </select>
                </label>
                <button type="button" className="secondary-btn full settings-field-full" onClick={()=>void (async()=>{
                  try{
                    const found=await fetchModels(generationDraft.llm);
                    setGenerationDraft(d=>({...d,llm:{...d.llm,model:d.llm.model || found[0] || ''}}));
                  }catch(e){setError(String(e));}
                })()}><RefreshCw size={13}/> GET MODELS</button>
                <label className="settings-field">
                  <span>PLANNER TEMPERATURE <b>{generationDraft.plannerTemperature.toFixed(2)}</b></span>
                  <input type="range" min={0} max={2} step={0.05} value={generationDraft.plannerTemperature} onChange={e=>setGenerationDraft(d=>({...d,plannerTemperature:Number(e.target.value)}))}/>
                  <small>Default: {defaultPlannerTemperature.toFixed(2)} · lower values make planning more deterministic</small>
                </label>
                <label className="settings-field">
                  <span>TAG GENERATOR TEMPERATURE <b>{generationDraft.tagTemperature.toFixed(2)}</b></span>
                  <input type="range" min={0} max={2} step={0.05} value={generationDraft.tagTemperature} onChange={e=>setGenerationDraft(d=>({...d,tagTemperature:Number(e.target.value)}))}/>
                  <small>Default: {defaultTagTemperature.toFixed(2)} · also used for validation repairs</small>
                </label>
                <label className="settings-field">
                  <span>BASE TEMPERATURE <b>{generationDraft.llm.temperature.toFixed(2)}</b></span>
                  <input type="range" min={0} max={2} step={0.05} value={generationDraft.llm.temperature} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,temperature:Number(e.target.value)}}))}/>
                  <small>Fallback/general LLM temperature kept for compatibility</small>
                </label>
                <label className="settings-field">
                  <span>MAX TOKENS</span>
                  <input type="number" min={128} max={16384} value={generationDraft.llm.maxTokens} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,maxTokens:Math.max(128,Number(e.target.value))}}))}/>
                </label>
                <label className="settings-field settings-field-full"><span>CONTEXT TOKENS</span><input type="number" min={2048} max={131072} step={1024} value={generationDraft.llm.contextTokens} onChange={e=>setGenerationDraft(d=>({...d,llm:{...d.llm,contextTokens:Math.max(2048,Number(e.target.value))}}))}/></label>
              </div>
            </section>
          </div>}

          {settingsTab==='system-prompts' && <div className="settings-tab-panel">
            <section className="settings-section settings-card">
              <div className="settings-section-title">ACTIVE CONTENT POLICY</div>
              <div className="settings-helper">The selected policy is inserted into all three stage system prompts at the CONTENT POLICY section. The policy text itself is editable.</div>
              <div className="demographic-grid settings-policy-grid">
                {(['safe','suggestive','explicit','no-limits'] as DemographicLevel[]).map(level=>
                  <button type="button" key={level} className={'demographic-card ' + (generationDraft.demographic===level ? 'active' : '')} onClick={()=>setGenerationDraft(d=>({...d,demographic:level}))}>
                    <b>{level.replace('-',' ').toUpperCase()}</b>
                    <span>{level==='safe' ? 'General audience' : level==='suggestive' ? 'Mature / suggestive' : level==='explicit' ? 'Adult explicit' : 'No added restriction'}</span>
                  </button>
                )}
              </div>
              <label className="settings-field settings-field-full">
                <span>{generationDraft.demographic.replace('-',' ').toUpperCase()} CONTENT POLICY</span>
                <textarea className="settings-textarea" value={generationDraft.demographicPrompts[generationDraft.demographic]} onChange={e=>setGenerationDraft(d=>({...d,demographicPrompts:{...d.demographicPrompts,[d.demographic]:e.target.value}}))}/>
              </label>
            </section>

            {([
              ['plannerSystemPrompt','PLANNER SYSTEM PROMPT','Plans the concrete visual decisions before tag generation.'],
              ['tagSystemPrompt','TAG GENERATION SYSTEM PROMPT','Converts the locked plan into the first final positive/negative tag pair.'],
              ['repairSystemPrompt','TAG REPAIR SYSTEM PROMPT','Repairs a failed tag pair using validation errors and the previous JSON.'],
            ] as const).map(([key,title,help])=>
              <section className="settings-section settings-card settings-stage-card" key={key}>
                <div className="settings-stage-head">
                  <div>
                    <div className="settings-section-title">{title}</div>
                    <div className="settings-helper">{help}</div>
                  </div>
                  <span className="settings-stage-badge">EDITABLE</span>
                </div>
                <label className="settings-field settings-field-full">
                  <span>SYSTEM PROMPT TEMPLATE</span>
                  <textarea className="settings-textarea system-prompt-editor" value={generationDraft[key]} onChange={e=>setGenerationDraft(d=>({...d,[key]:e.target.value}))}/>
                </label>
                <div className="settings-helper">{'Placeholders: {{DEMOGRAPHIC_POLICY}}, {{MIN_POSITIVE_TAGS}}, {{MIN_NEGATIVE_TAGS}}, {{MAX_NEGATIVE_TAGS}}, {{MAX_TAG_LENGTH}}'}</div>
                <label className="settings-field settings-field-full">
                  <span>EFFECTIVE SYSTEM PROMPT · SENT TO MODEL</span>
                  <textarea
                    className="settings-textarea system-prompt-preview"
                    value={renderPromptTemplate(generationDraft[key],{
                      DEMOGRAPHIC_POLICY:generationDraft.demographicPrompts[generationDraft.demographic],
                      MIN_POSITIVE_TAGS:String(generationDraft.minPositiveTags),
                      MIN_NEGATIVE_TAGS:String(generationDraft.minNegativeTags),
                      MAX_NEGATIVE_TAGS:String(generationDraft.maxNegativeTags),
                      MAX_TAG_LENGTH:String(generationDraft.maxTagLength),
                    })}
                    readOnly
                  />
                </label>
              </section>
            )}
          </div>}

          {settingsTab==='prompt-templates' && <div className="settings-tab-panel">
            <section className="settings-section settings-card">
              <div className="settings-section-title">TAG GENERATION USER PROMPT</div>
              <div className="settings-helper">This is the user message sent with the tag-generation system prompt. Checkpoint, compatibility and LoRA data are inserted at generation time.</div>
              <label className="settings-field settings-field-full">
                <span>PRIMARY USER PROMPT TEMPLATE · EDITABLE</span>
                <textarea className="settings-textarea prompt-template-editor" value={generationDraft.userPromptTemplate} onChange={e=>setGenerationDraft(d=>({...d,userPromptTemplate:e.target.value}))}/>
              </label>
              <div className="settings-helper">{'Placeholders: {{CHECKPOINT}}, {{BASE}}, {{COMPATIBILITY}}, {{LORA_METADATA}}, {{CHARACTER}}, {{SETTING}}, {{POSE}}, {{EXPRESSION}}, {{DRESS}}, {{COMPOSITION}}, {{EXTRA}}'}</div>
            </section>

            <section className="settings-section settings-card">
              <div className="settings-section-title">TAG REPAIR USER PROMPT</div>
              <div className="settings-helper">This is the user message sent only when validation fails. The app appends the actual validation errors and previous JSON after this template.</div>
              <label className="settings-field settings-field-full">
                <span>REPAIR USER PROMPT TEMPLATE · EDITABLE</span>
                <textarea className="settings-textarea prompt-template-editor" value={generationDraft.expansionPromptTemplate} onChange={e=>setGenerationDraft(d=>({...d,expansionPromptTemplate:e.target.value}))}/>
              </label>
              <div className="settings-helper">{'Placeholder: {{PREVIOUS_JSON}} plus the same scene/model placeholders as the primary template.'}</div>
            </section>
          </div>}

          {settingsTab==='scene' && <div className="settings-tab-panel">
            <section className="settings-section settings-card">
              <div className="settings-section-title">SCENE CONSTRAINTS</div>
              <div className="settings-form-grid">
                {(['setting','pose','expression','character','dress','composition'] as const).map(key=>
                  <label className="settings-field" key={key}>
                    <span>{key.replace('_',' ').toUpperCase()}</span>
                    <input value={generationDraft.constraints[key]} onChange={e=>setGenerationDraft(d=>({...d,constraints:{...d.constraints,[key]:e.target.value}}))} placeholder={key==='character' ? 'Compatible character LoRA' : 'Random if blank'}/>
                  </label>
                )}
                <label className="settings-field settings-field-full">
                  <span>ADDITIONAL CONSTRAINTS</span>
                  <textarea className="settings-textarea" value={generationDraft.constraints.additional} onChange={e=>setGenerationDraft(d=>({...d,constraints:{...d.constraints,additional:e.target.value}}))} placeholder="Additional visual constraints"/>
                </label>
              </div>
            </section>
          </div>}

          {settingsTab==='output' && <div className="settings-tab-panel">
            <section className="settings-section settings-card">
              <div className="settings-section-title">TAG VALIDATION</div>
              <div className="settings-helper">Configure tag-count limits and the maximum character length for each ordinary tag before validation rejects it. Exact documented activation prompts remain exempt.</div>
              <div className="settings-form-grid">
                <label className="settings-field settings-number-field">
                  <span>MINIMUM POSITIVE TAGS</span>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    step={1}
                    value={generationDraft.minPositiveTags}
                    onChange={e=>setGenerationDraft(d=>({...d,minPositiveTags:Math.max(1,Math.min(100,Number(e.target.value)||1))}))}
                  />
                  <small>Default: {defaultMinPositiveTags}</small>
                </label>
                <label className="settings-field settings-number-field">
                  <span>MINIMUM NEGATIVE TAGS</span>
                  <input
                    type="number"
                    min={1}
                    max={generationDraft.maxNegativeTags}
                    step={1}
                    value={generationDraft.minNegativeTags}
                    onChange={e=>setGenerationDraft(d=>({...d,minNegativeTags:Math.max(1,Math.min(d.maxNegativeTags,Number(e.target.value)||1))}))}
                  />
                  <small>Default: {defaultMinNegativeTags} · must not exceed maximum</small>
                </label>
                <label className="settings-field settings-number-field">
                  <span>MAXIMUM NEGATIVE TAGS</span>
                  <input
                    type="number"
                    min={1}
                    max={absoluteMaxTagLimit}
                    step={1}
                    value={generationDraft.maxNegativeTags}
                    onChange={e=>{
                      const next=Math.max(1,Math.min(absoluteMaxTagLimit,Number(e.target.value)||1));
                      setGenerationDraft(d=>({...d,maxNegativeTags:next,minNegativeTags:Math.min(d.minNegativeTags,next)}));
                    }}
                  />
                  <small>Default: {defaultMaxNegativeTags} · absolute maximum: {absoluteMaxTagLimit}</small>
                </label>
                <label className="settings-field settings-number-field">
                  <span>MAXIMUM TAG LENGTH · CHARACTERS</span>
                  <input
                    type="number"
                    min={1}
                    max={absoluteMaxTagLength}
                    step={1}
                    value={generationDraft.maxTagLength}
                    onChange={e=>setGenerationDraft(d=>({...d,maxTagLength:Math.max(1,Math.min(absoluteMaxTagLength,Number(e.target.value)||1))}))}
                  />
                  <small>Default: {defaultMaxTagLength} · absolute maximum: {absoluteMaxTagLength}</small>
                </label>
                <label className="settings-field settings-number-field">
                  <span>TAG GENERATION RETRIES</span>
                  <input
                    type="number"
                    min={0}
                    max={absoluteMaxTagGenerationRetries}
                    step={1}
                    value={generationDraft.tagGenerationRetries}
                    onChange={e=>setGenerationDraft(d=>({...d,tagGenerationRetries:Math.max(0,Math.min(absoluteMaxTagGenerationRetries,Number(e.target.value)||0))}))}
                  />
                  <small>Default: {defaultTagGenerationRetries} · number of repair calls after the initial tag generation</small>
                </label>
              </div>
            </section>

            <section className="settings-section settings-card">
              <div className="settings-section-title">LoRA SELECTION LIMITS</div>
              <div className="settings-helper">MAX CHARACTER LoRAs controls how many character-identity LoRAs can be manually selected or included by RANDOMIZE LORAS and AUTO GENERATE.</div>
              <div className="settings-form-grid">
                <label className="settings-field"><span>MAX LoRAs</span><input type="number" min={1} max={absoluteMaxLoraLimit} value={generationDraft.maxLoras} onChange={e=>setGenerationDraft(d=>({...d,maxLoras:Math.max(1,Number(e.target.value))}))}/></label>
                <label className="settings-field"><span>MAX CHARACTER LoRAs</span><input type="number" min={1} max={Math.min(absoluteMaxLoraLimit,generationDraft.maxLoras)} value={generationDraft.maxCharacterLoras} onChange={e=>setGenerationDraft(d=>({...d,maxCharacterLoras:Math.max(1,Math.min(d.maxLoras,Number(e.target.value)||1))}))}/></label>
                <label className="settings-field"><span>RANDOM MIN</span><input type="number" min={1} max={16} value={generationDraft.randomLoraMin} onChange={e=>setGenerationDraft(d=>({...d,randomLoraMin:Math.max(1,Number(e.target.value))}))}/></label>
                <label className="settings-field"><span>RANDOM MAX</span><input type="number" min={1} max={16} value={generationDraft.randomLoraMax} onChange={e=>setGenerationDraft(d=>({...d,randomLoraMax:Math.max(1,Number(e.target.value))}))}/></label>
              </div>
            </section>

            <section className="settings-section settings-card">
              <div className="settings-section-title">IMAGE SAMPLING</div>
              <div className="settings-form-grid">
                <label className="settings-field"><span>WIDTH</span><input type="number" min={64} step={64} value={generationDraft.width} onChange={e=>setGenerationDraft(d=>({...d,width:Number(e.target.value)}))}/></label>
                <label className="settings-field"><span>HEIGHT</span><input type="number" min={64} step={64} value={generationDraft.height} onChange={e=>setGenerationDraft(d=>({...d,height:Number(e.target.value)}))}/></label>
                <label className="settings-field"><span>STEPS</span><input type="number" min={1} max={200} value={generationDraft.steps} onChange={e=>setGenerationDraft(d=>({...d,steps:Number(e.target.value)}))}/></label>
                <label className="settings-field"><span>CFG</span><input type="number" min={0} step={0.1} value={generationDraft.cfg} onChange={e=>setGenerationDraft(d=>({...d,cfg:Number(e.target.value)}))}/></label>
                <label className="settings-field settings-field-full"><span>SAMPLER</span><input value={generationDraft.sampler} onChange={e=>setGenerationDraft(d=>({...d,sampler:e.target.value}))}/></label>
              </div>
            </section>
          </div>}
        </div>

        <div className="drawer-foot">
          <span>SETTINGS AUTO-SAVED</span>
          <button className="primary-btn" onClick={()=>setSettingsOpen(false)}>DONE</button>
        </div>
        </div>
      </div>}
  </div>
}

export default App;
