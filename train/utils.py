import torch

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["lm_head"]):
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names



def vlm_img_pipeline_gradient_config(model, training_args, config):
    # When using LoRA, the model is rapped once more.
    if config.use_lora:
        vision_tower = model.model.model.vision_embed_tokens.img_processor.vision_model
        vision_tower.to(device=training_args.device)

        if not config.train.img_projector:
            for p in model.model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = False
        else:
            for p in model.model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = True

        if not config.train.vision_tower:
            for p in model.model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = False
        else:
            for p in model.model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = True

    else:
        vision_tower = model.model.vision_embed_tokens.img_processor.vision_model
        vision_tower.to(device=training_args.device)

        if not config.train.img_projector:
            for p in model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = False
        else:
            for p in model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = True

        if not config.train.vision_tower:
            for p in model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = False
        else:
            for p in model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = True