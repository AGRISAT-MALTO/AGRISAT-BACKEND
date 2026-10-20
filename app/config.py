from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    gee_service_account_key: str = "{}"
    hf_model_url: str = ""
    field_segmentation_model_url: str = "http://localhost:8000"
    vhr_model_url: str = "http://localhost:8010"
    # Auto-start du service de tracage VHR (serve_vhr.py, modele delineate_jecam_complet.pt).
    # vhr_service_dir vide : pas d'auto-start (le service doit etre lance a la main).
    # vhr_service_python vide : utilise le venv du service s'il existe, sinon le python courant.
    vhr_service_dir: str = ""
    vhr_service_python: str = ""
    vhr_service_port: int = 8010
    # Cache disque des zones tracees par le service VHR (vide : AGRISAT-BACKEND/.cache/vhr_parcels).
    vhr_cache_dir: str = ""
    vhr_cache_ttl_days: float = 30.0
    google_maps_api_key: str = ""
    airbus_oneatlas_api_key: str = ""
    port: int = 3001
    host: str = "0.0.0.0"
    frontend_origin: str = ""
    planet_api_key: str = ""


settings = Settings()
