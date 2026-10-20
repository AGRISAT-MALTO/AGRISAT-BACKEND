from app.services.automatic_parcels import _is_plausible_field_shape


def _polygon(points):
    return [{"lat": lat, "lng": lng} for lat, lng in points]


def test_square_field_shape_is_kept():
    polygon = _polygon([
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
    ])
    assert _is_plausible_field_shape(polygon, 1_000_000) is True


def test_spiky_shape_is_rejected():
    polygon = _polygon([
        (0.0, 0.0),
        (0.2, 0.0),
        (0.4, 0.8),
        (0.8, 0.1),
        (1.0, 0.0),
        (0.9, 1.0),
        (0.5, 1.5),
        (0.2, 1.0),
        (0.0, 0.9),
        (0.0, 0.0),
    ])
    assert _is_plausible_field_shape(polygon, 1_000_000) is False
