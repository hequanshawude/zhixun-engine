"""PyTRiO 官方登录凭据与现有 chat/completions 客户端的桥接。"""
import os

BASE_URL = "https://pytrio.com/api/openai/v1"
MODEL = "Qwen/Qwen3.8-27B"


def configure_credentials(cfg):
    # 自动读取的账号凭据只允许发送到已核对的官方端点。
    if cfg.base_url.rstrip('/') != BASE_URL:
        raise ValueError('PyTRiO 登录凭据只能用于官方模型接口')
    key = os.getenv('HARNESS_API_KEY') or os.getenv('PYTRIO_API_KEY')
    if not key:
        try:
            # 与 requirements-pytrio.txt 锁定的 SDK 0.2.9 配套。
            from pytrio.config.config import auth_state, configuration
        except ImportError:
            raise ValueError('请先安装 requirements-pytrio.txt 或注入 PYTRIO_API_KEY') from None
        if configuration.base_url.rstrip('/') != 'https://pytrio.com':
            raise ValueError('当前 PyTRiO 登录配置不是 pytrio.com，请先切换官方服务')
        if auth_state.config_error:
            raise ValueError('PyTRiO 本地登录配置无法读取，请重新登录')
        key = auth_state.api_key
    if not key or not key.strip():
        raise ValueError('尚未登录 PyTRiO，请运行 login-pytrio.cmd；不要把密钥发到聊天中')
    cfg.api_key = key.strip()

