from positronic.gui.eval import CONTENT_SIZE, MIN_UI_SCALE, fit_ui_scale

ROOMY = (CONTENT_SIZE[0] * 10, CONTENT_SIZE[1] * 10)


def fits(scale: float, viewport: tuple[int, int]) -> bool:
    return all(need * scale <= avail for avail, need in zip(viewport, CONTENT_SIZE, strict=True))


def test_scale_drops_to_what_the_viewport_holds():
    viewport = (CONTENT_SIZE[0] * 2, CONTENT_SIZE[1] * 2)

    assert fit_ui_scale(3.0, viewport) == 2.0
    assert fits(fit_ui_scale(3.0, viewport), viewport)


def test_scale_follows_the_axis_that_binds():
    wide = (CONTENT_SIZE[0] * 8, CONTENT_SIZE[1] * 2)
    tall = (CONTENT_SIZE[0] * 2, CONTENT_SIZE[1] * 8)

    assert fit_ui_scale(4.0, wide) == 2.0
    assert fit_ui_scale(4.0, tall) == 2.0


def test_a_viewport_with_room_leaves_the_requested_scale_alone():
    assert fit_ui_scale(3.0, ROOMY) == 3.0


def test_a_viewport_that_exactly_holds_the_request_leaves_it_alone():
    exact = (CONTENT_SIZE[0] * 3, CONTENT_SIZE[1] * 3)

    assert fit_ui_scale(3.0, exact) == 3.0


def test_the_floor_holds_on_a_viewport_nothing_fits():
    assert fit_ui_scale(3.0, (10, 10)) == MIN_UI_SCALE


def test_a_scale_below_the_floor_is_never_raised_to_it():
    assert fit_ui_scale(0.5, ROOMY) == 0.5
    assert fit_ui_scale(0.5, (10, 10)) == 0.5
