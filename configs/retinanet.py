
def retinanet_config(args):

    args.add_argument('--anchor_areas', type=str, default=[32*32., 64*64., 128*128., 256*256., 512*512.])
    args.add_argument('--aspect_ratios', type=str, default=(1/2., 1/1., 2/1.))
    args.add_argument('--scale_ratios', type=str, default=(1., pow(2, 1/3), pow(2, 2/3.)))
    args.add_argument('--img_size', type=int, default=512)

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