"""Package entrypoint for the motor SAC + EC/CB experiment."""


def main(**kwargs):
    from . import _mc_sac_ec_cb_impl as impl

    if kwargs:
        return impl.main(**kwargs)

    args = impl._parse_args()
    return impl.main(
        seeds=args.seeds,
        use_pretrained_ec_cb=args.use_pretrained_ec_cb,
        freeze_pretrained_ec_cb=args.freeze_pretrained_ec_cb,
        ec_cb_ckpt=args.ec_cb_ckpt,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()
