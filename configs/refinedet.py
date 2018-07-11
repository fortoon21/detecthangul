
def refinedet_config(args):

    args.add_argument('--feature_maps', type=str, default=[40, 20, 10, 5])
    args.add_argument('--aspect_ratios', type=str, default=[[2], [2], [2], [2]])
    args.add_argument('--min_dim', type=int, default=320)
    args.add_argument('--steps', type=str, default=[8, 16, 32, 64])
    args.add_argument('--min_sizes', type=str, default=[32, 64, 128, 256])
    args.add_argument('--max_sizes', type=str, default=[])
    args.add_argument('--variance', type=str, default=[0.1, 0.2])
    args.add_argument('--clip', type=bool, default=True)
    args.add_argument('--img_size', type=int, default=320)

    args.add_argument('--use_pretrained', type=bool, default=True)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=1e-4)
    args.add_argument('--lr_steps', type=str, default=[240, 280])
    args.add_argument('--lr_decay_rate', type=float, default=0.1)
    args.add_argument('--accum_grad', type=int, default=4)

    args.add_argument('--fold', type=int, default=9)

    opt = args.parse_args()

    return opt