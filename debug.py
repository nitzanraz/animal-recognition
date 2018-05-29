import env
args = env.get_parser().parse_args()
args.val_dir = '/home/eran/Pictures/myimagenet/val'
args.train_dir = '/home/eran/Pictures/myimagenet/train'
args.use_gpu = True
e = env.Environment(args)
