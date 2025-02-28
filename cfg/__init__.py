from hydra_zen import ZenStore, make_config as hz_make_config, builds as hz_builds


class IronicZenStore(ZenStore):
    def __init__(self, *args, **kwargs):
        """
        ZenStore wrapper that is used to enable automatic defaults configuration.
        """
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)

        res.__store_name__ = self._defaults['group']
        res.__varname__ = kwargs.get('name', None)

        return res


def make_config(*args, **kwargs):
    """
    make_config wrapper that is used to enable automatic defaults configuration.

    Example:
        >>> env_store = cfg.store(group="env")
        >>> cfg.env.umi_env = env_store(cfg.env.umi_env(), name="umi")
        >>> ui_store = cfg.store(group="ui")
        >>> cfg.ui.teleop_ui = ui_store(cfg.ui.teleop_ui(cfg.ui.teleop), name="teleop_ui")

        >>> hydra_zen.make_config(
        >>>     env=cfg.env.umi_env,
        >>>     ui=cfg.ui.teleop_ui,
        >>>     hydra_defaults=["_self_", {"env": "umi"}, {"ui": "teleop_ui"}]
        >>> )
        >>> # Equivalent to:
        >>> cfg.make_config(env=cfg.env.umi_env, ui=cfg.ui.teleop_ui)
    """
    assert 'hydra_defaults' not in kwargs, "hydra_defaults will be set automatically"
    hydra_defaults = ['_self_']

    for k, v in kwargs.items():
        if hasattr(v, '__store_name__'):
            if v.__store_name__ != k:
                hydra_defaults.append({f"/{v.__store_name__}@{k}": v.__varname__})
            else:
                hydra_defaults.append({f"/{v.__store_name__}": v.__varname__})
            kwargs[k] = None
    if len(hydra_defaults) > 1:
        kwargs['hydra_defaults'] = hydra_defaults

    res = hz_make_config(*args, **kwargs,)
    return res


def builds(*args, **kwargs):
    """
    Builds wrapper that is used to enable automatic defaults configuration.

    Also changes default value for populate_full_signature to True.

    Example:
        >>> def combine_systems(env, ui):
        >>>     ...
        >>> cfg.combined_system = hydra_zen.builds(
        >>>     combine_systems,
        >>>     env=cfg.env.umi_env,
        >>>     ui=cfg.ui.teleop_ui,
        >>>     hydra_defaults=["_self_", {"env": "umi"}, {"ui": "teleop_ui"}],
        >>>     populate_full_signature=True,
        >>> )
        >>> # Equivalent to:
        >>> cfg.combined_system = cfg.builds(combine_systems, env=cfg.env.umi_env, ui=cfg.ui.teleop_ui)
    """
    assert 'hydra_defaults' not in kwargs, "hydra_defaults will be set automatically"
    hydra_defaults = ['_self_']

    if 'populate_full_signature' not in kwargs:
        kwargs['populate_full_signature'] = True

    for k, v in kwargs.items():
        if hasattr(v, '__store_name__'):
            if v.__store_name__ != k:
                hydra_defaults.append({f"/{v.__store_name__}@{k}": v.__varname__})
            else:
                hydra_defaults.append({f"/{v.__store_name__}": v.__varname__})
            kwargs[k] = None

    if len(hydra_defaults) > 1:
        kwargs['hydra_defaults'] = hydra_defaults

    res = hz_builds(*args, **kwargs)
    return res


store: IronicZenStore = IronicZenStore(
    name="zen_store",
    deferred_to_config=True,
    deferred_hydra_store=True,
)
