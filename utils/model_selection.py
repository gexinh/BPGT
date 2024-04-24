"""
    Choose Model based on args
                                        Geneformer   GCN    graph transformer   MLP multi-label
    deephis: resnet + mlp 
    genehe: resnet + self-attention
    abmil: dino + abmil
    hipt: dino + hierachical transformer
    transMIL: dino + cor mil
    KAT: dino + Kernel transformer
"""

def model_selection(args):
    # transfer multi-label classification

    if args.model == 'deephis':
        args.agg_method = 'dense'
        args.model_size = '1024_small'
        args.pretrained_model = 'resnet'

    elif args.model == 'genehe':
        args.agg_method = 'self_att'
        args.model_size = '1024_small'
        args.pretrained_model = 'resnet'

    elif args.model == 'abmil':
        args.agg_method = 'abmil'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'
        
    elif args.model == 'hipt':
        args.agg_method = 'hipt'
        args.model_size = 'hipt_att'
        args.pretrained_model = 'hipt'

    elif args.model == 'trans_mil':
        args.agg_method = 'trans_mil'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'
        args.region_cls_model = 'corr'

    elif args.model == 'kat':
        args.agg_method = 'kat'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'

    return args

def module_selection(args, model_dict):
    if args.graph_module == 'geneformer':
        """ construct the graph 
        """
        from .graph_utils import mutual_info_encoding, organ_positional_encoding
        pe = organ_positional_encoding(args, df=args.df) 
        mask = mutual_info_encoding(args, df=args.df, if_cls_token= args.cls_token)        
        model_dict["pe"]=pe
        model_dict["spatial_mask"] = mask

    elif args.graph_module == 'gcn':
        from .graph_utils import mutual_info_encoding, organ_positional_encoding
        pe = organ_positional_encoding(args, df=args.df) 
        mask = mutual_info_encoding(args, df=args.df, if_cls_token= args.cls_token)        
        model_dict["pe"]=pe
        model_dict["spatial_mask"] = mask

        model_dict["adj"] = args.adj + pe + mask

    elif args.graph_module == 'graph_transformer':
        from .graph_utils import mutual_info_encoding, organ_positional_encoding
        pe = organ_positional_encoding(args, df=args.df) 
        mask = mutual_info_encoding(args, df=args.df, if_cls_token= args.cls_token)        
        model_dict["pe"]=pe
        model_dict["spatial_mask"] = mask

        model_dict["adj"] = args.adj + mask + pe 

    elif args.graph_module == 'mlp':
        pass
    elif args.graph_module == 'mcat':
        pass

    return model_dict