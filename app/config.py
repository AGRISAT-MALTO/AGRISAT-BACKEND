from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    gee_service_account_key: str = "{}"
    hf_model_url: str = ""
    field_segmentation_model_url: str = "http://localhost:8000"
    google_maps_api_key: str = ""
    airbus_oneatlas_api_key: str = ""
    port: int = 3001
    host: str = "0.0.0.0"
    frontend_origin: str = ""
    planet_api_key: str = ""


settings = Settings()
