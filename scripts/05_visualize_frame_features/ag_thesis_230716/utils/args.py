def arg_dict_to_list(args):
    if args is None:
        return None
    pairs = [(f"--{k}", str(v)) for k, v in args.items()]
    return list([entry for pair in pairs for entry in pair])
