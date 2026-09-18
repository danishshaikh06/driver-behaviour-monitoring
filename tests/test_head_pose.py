from driver_monitor.config import settings


def test_person_model_default_is_configured():
    assert settings.person_model.endswith(".pt")
