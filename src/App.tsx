
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

    // Random selection is intentionally strict: compatibleLoras only contains
    // LoRAs carrying the checkpoint's exact base-model tag (e.g. "anima").
    // Registry compatibility is then used as the final authority. If a tagged
    // candidate is still rejected, discard it and try another matching LoRA.
    const failedRandomIds=new Set<string>();
    const maxAttempts=Math.min(
      8,
      Math.max(1,compatibleLoras.filter(lora=>!manualIds.includes(lora.id)).length),
    );
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
      randomIds.push(
        ...availablePool
          .slice(0,remainingSlots)
          .map(lora=>lora.id),
      );

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
        // Manual LoRAs are user-selected and should not be silently replaced.
        // Only discard automatically randomized candidates.
        for(const id of randomIds) failedRandomIds.add(id);
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