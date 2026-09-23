export type ProviderKind = 'ollama' | 'openai-compatible';

export interface ModelInfo {
  id: string;
  name: string;
  type: 'checkpoint' | 'lora';
  path: string;
  size: number;
  baseModel?: string;
  tags: string[];
  activationTags: string[];
  character: boolean;
  thumbnail?: string;
  source: 'comfyui' | 'raphael-cache' | 'merged';
  cacheName?: string;
  cacheDescription?: string;
}

export interface LibrarySnapshot {
  checkpoints: ModelInfo[];
  loras: ModelInfo[];
  sourceRoots: string[];
  warnings: string[];
}

export interface LlmSettings {
  provider: ProviderKind;
  baseUrl: string;
  apiKey: string;
  model: string;
  temperature: number;
  maxTokens: number;
}

export interface Constraints {
  setting: string;
  pose: string;
  expression: string;
  character: string;
  dress: string;
  composition: string;
  additional: string;
  randomLoraMin: number;
  randomLoraMax: number;
}

export interface SceneSelection {
  setting: string;
  pose: string;
  expression: string;
  character: string;
  dress: string;
  composition: string;
}

export interface SelectedLora {
  id: string;
  name: string;
  path: string;
  weight: number;
  activationTags: string[];
  tags: string[];
  description?: string;
  character: boolean;
  baseModel?: string;
}

export interface PreparedGeneration {
  checkpoint: ModelInfo;
  loras: SelectedLora[];
  scene: SceneSelection;
  compatibilityKeys: string[];
}

export interface PromptPair {
  positive_prompt: string;
  negative_prompt: string;
  rationale?: string;
}

export interface GenerationRecord {
  id: string;
  timestamp: string;
  provider: ProviderKind;
  model: string;
  checkpoint: ModelInfo;
  loras: SelectedLora[];
  scene: SceneSelection;
  positivePrompt: string;
  negativePrompt: string;
  rationale?: string;
  generationSettings?: GenerationSettings;
  imageDataUrl?: string;
  imageFilename?: string;
  workflow: unknown;
  comfyPromptId?: string;
}


export type DemographicLevel = 'safe' | 'suggestive' | 'explicit' | 'no-limits';

export interface DemographicPrompts {
  safe: string;
  suggestive: string;
  explicit: string;
  'no-limits': string;
}

export interface GenerationSettings {
  llm: LlmSettings;
  systemPrompt: string;
  demographic: DemographicLevel;
  demographicPrompts: DemographicPrompts;
  maxLoras: number;
  randomLoraMin: number;
  randomLoraMax: number;
  constraints: Constraints;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
}

export interface WebHostInfo {
  running: boolean;
  port: number;
  localUrl: string;
  lanUrl: string;
}
