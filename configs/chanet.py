
def chanet_config(args):

    args.add_argument('--base_model', type=str, default='resnet50')
    args.add_argument('--img_size', type=int, default=96)

    # v_caption
    args.add_argument('--config_class_num', type=int, default=6)
    args.add_argument('--first_class_num',  type=int, default=19)
    args.add_argument('--middle_class_num', type=int, default=21)
    args.add_argument('--last_class_num',   type=int, default=28)

    args.add_argument('--use_pretrained', type=bool, default=False)

    args.add_argument('--lr', type=float, default=1e-2)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=1e-4)
    args.add_argument('--accum_grad', type=int, default=1)

    opt = args.parse_args()

    return opt