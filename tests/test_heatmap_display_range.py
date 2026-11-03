"""Explicit display limits reach preparation, meshes, and every map API."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hornlab_plots import _heatmap as heatmap

FREQUENCIES = [100.0, 200.0]
ANGLES = [0.0, 90.0]
VALUES = np.array([[0.0, 0.0], [-35.0, -50.0]])
DIRECTIVITY = {
    "horizontal": [
        [[angle, value] for angle, value in zip(ANGLES, column)]
        for column in VALUES.T
    ],
}


@pytest.mark.parametrize("reference_level", [-3.0, -6.0, -9.0, -12.0])
def test_display_range_reaches_preparation_and_mesh(reference_level):
    floor = reference_level * 5
    before = VALUES.copy()
    angles, freqs, values = heatmap.prepare_heatmap_data(
        ANGLES, FREQUENCIES, VALUES, display_range_db=(floor, 0.0),
    )
    np.testing.assert_array_equal(values[-1, [0, -1]], np.clip([-35., -50.], floor, 0.))
    np.testing.assert_array_equal(VALUES, before)
    fig, ax = plt.subplots()
    try:
        heatmap.render_single_heatmap(
            ax, freqs, angles, values, "Directivity", reference_level=reference_level,
            display_range_db=(floor, 0.0),
        )
        assert ax.collections[0].get_clim() == (floor, 0.0)
        assert fig.axes[1].get_ylim() == (floor, 0.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("api", ["legacy", "array", "file"])
@pytest.mark.parametrize("reference_level", [-3.0, -6.0, -9.0, -12.0])
def test_public_map_apis_forward_display_range(monkeypatch, tmp_path, api, reference_level):
    captured = []
    prepare = heatmap.prepare_heatmap_data
    render = heatmap.render_single_heatmap
    display_range = (reference_level * 5, 0.0)

    def capture_prepare(*args, **kwargs):
        assert kwargs["display_range_db"] == display_range
        captured.append("prepare")
        return prepare(*args, **kwargs)

    def capture_render(*args, **kwargs):
        assert kwargs["display_range_db"] == display_range
        captured.append("render")
        return render(*args, **kwargs)

    monkeypatch.setattr(heatmap, "prepare_heatmap_data", capture_prepare)
    monkeypatch.setattr(heatmap, "render_single_heatmap", capture_render)
    kwargs = {"reference_level": reference_level, "display_range_db": display_range, "dpi": 30}
    if api == "legacy":
        result = heatmap.directivity_heatmap_from_legacy_dict(
            FREQUENCIES, DIRECTIVITY,
            reference_frequencies=FREQUENCIES,
            reference_directivity=DIRECTIVITY,
            **kwargs,
        )
        assert captured == ["prepare", "prepare", "render"]
    elif api == "array":
        result = heatmap.directivity_heatmap_b64(FREQUENCIES, ANGLES, VALUES, **kwargs)
        assert captured == ["prepare", "render"]
    else:
        result = heatmap.save_directivity_plot(
            tmp_path / "map.png", FREQUENCIES, DIRECTIVITY, **kwargs,
        )
        assert captured == ["prepare", "render"]
    assert result


def test_missing_map_rows_use_the_selected_floor():
    _, _, values = heatmap.prepare_heatmap_data(
        ANGLES, FREQUENCIES, np.full((2, 2), np.nan), display_range_db=(-60., 0.),
    )
    np.testing.assert_array_equal(values, -60.)


@pytest.mark.parametrize("limits", [(0., -30.), (-30., -30.), (np.nan, 0.), (-30., np.inf)])
def test_invalid_display_ranges_give_a_remedy(limits):
    with pytest.raises(ValueError, match="display_range_db.*increasing.*-60"):
        heatmap.prepare_heatmap_data(ANGLES, FREQUENCIES, VALUES, display_range_db=limits)


def test_default_display_range_is_unchanged():
    implicit = heatmap.prepare_heatmap_data(ANGLES, FREQUENCIES, VALUES)
    explicit = heatmap.prepare_heatmap_data(
        ANGLES, FREQUENCIES, VALUES, display_range_db=(-30., 0.),
    )
    for first, second in zip(implicit, explicit):
        np.testing.assert_array_equal(first, second)
