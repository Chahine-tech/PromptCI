from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-6"
    default_timeout: float = 30.0
    default_max_tokens: int = 1024
    snapshot_dir: str = ".pytest-prompts/snapshots"


settings = Settings()
