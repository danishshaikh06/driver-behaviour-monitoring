from driver_monitor.config import settings


def test_phone_model_default_is_configured():
    assert settings.phone_model.endswith(".pt")
