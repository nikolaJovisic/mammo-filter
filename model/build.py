from . import vision_transformer as vits

def build_model(args):
    vit_kwargs = dict(
        img_size=args.img_size,
        patch_size=args.patch_size,
        init_values=args.layerscale,
        ffn_layer=args.ffn_layer,
        block_chunks=args.block_chunks,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        num_register_tokens=args.num_register_tokens,
        interpolate_offset=args.interpolate_offset,
        interpolate_antialias=args.interpolate_antialias,
    )
    student = vits.__dict__[args.arch](
        **vit_kwargs,
        drop_path_rate=args.drop_path_rate,
        drop_path_uniform=args.drop_path_uniform,
    )
    return student