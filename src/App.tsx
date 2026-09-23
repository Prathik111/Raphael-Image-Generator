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