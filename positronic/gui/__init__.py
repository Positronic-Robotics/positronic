import pimm


def dpg_ui() -> pimm.ControlSystem:
    """Builds the Dearpygui desktop UI, importing `dearpygui` only when one is asked for."""
    from positronic.gui.dpg import DearpyguiUi  # noqa: PLC0415

    return DearpyguiUi()
