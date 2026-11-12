# PyTRiO 本机接入

官方文档：[采样 SDK](https://docs.pytrio.com/docs/guide/sample)、[OpenAI 兼容接口](https://docs.pytrio.com/docs/advanced/openai)。当前 27B 的完整本机对照见[Qwen27B 真实数据对照](Qwen27B真实数据对照.md)。

本机可选依赖安装：`.venv\Scripts\python.exe -m pip install -r requirements-pytrio.txt`。
官方 SDK 固定为 0.2.9，凭据桥接与该版本配套；升级需重新验证。

## 登录与连接检查

双击项目根目录 `login-pytrio.cmd`，从 https://pytrio.com/dashboard 复制 API Key，在官方客户端提示处粘贴并回车。密钥不显示在输入行，由 SDK 保存在用户目录 `~/.pytrio`，不要上传该目录。

登录后窗口自动执行 `scripts/pytrio_check.py`：读取账号可用模型列表，核对 `Qwen/Qwen3.8-27B`，不启动训练或推理。检查结果为 `out/pytrio-setup/connection.json`。未取得 `connected` 状态前，不表示账号连接已验收。

单独复查：`.venv\Scripts\python.exe scripts\pytrio_check.py`。

## 项目模型配置

连接配置：`configs/pytrio.yaml`；Gas 完整对照配置：`configs/gas-pytrio-27b.yaml`。

- 服务地址：`https://pytrio.com/api/openai/v1`。
- 大小模型路由均使用 `Qwen/Qwen3.8-27B`。
- 从 `HARNESS_API_KEY`、`PYTRIO_API_KEY` 或官方 SDK 当前登录状态依次读取凭据，实际密钥不写入 YAML。
- 自动读取的凭据仅允许发往上述官方地址；其他地址报错停止。
- 策略运行使用 PyTRiO 官方采样 SDK，并通过分词器关闭思考模式；短连接检查仍使用 OpenAI 兼容接口。两者调用的都是 PyTRiO 千问模型。
- 普通默认配置仍关闭大模型；启动真实模型策略时必须显式指定此配置及 `--policy llm`。
- Gas 27B 配置单次最多输出 128 token，总策略预算 10,000 token；请求前保守预留输入字节数、最多输出量和余量，余额不足时回退本地规则。此预算不是人民币余额硬上限；服务端计量变化和未知用量仍需监控。
- Gas 27B 实验使用 SwanLab 在线日志，失败降级也保留本地记录。

## 可选真实推理验证（消耗平台额度）

`.venv\Scripts\python.exe scripts\pytrio_check.py --smoke`

先核对模型列表，再发起一次短提示调用，最多输出 64 token，无重试，保存用量与非空文本检查结果到 `out/pytrio-setup/smoke.json`，不保存密钥和回答原文。这个检查只验证基础模型通信，不代表实际策略 JSON 输出或比赛效果已经通过。

正式评测网络是否允许调用 PyTRiO 服务、额度和资源限制仍需核对。当前 `zhixun-engine:qwen27b-preflight` 已包含 SDK 和 27B 分词器，尚非官方验收镜像。

## 2026-11-10 实测状态

- 本机官方 SDK 登录成功，账号连接检查为 `connected`；模型列表包含 `Qwen/Qwen3.8-27B`。
- 充值前一次短消息推理返回 HTTP 409 / `insufficient_quota`，记录为 `quota_blocked`；用户充值后短消息验证为 `inference_verified`，59 输入 token、30 输出 token。
- 首次四批小规模策略演练用 384 输出 token，三次调用均被千问思考过程耗尽，返回空正文并降级规则；平均分比静态模型低 0.0207。复查发现服务完成原因 `length`，输出 384 token 全是思考 token。
- 单次上限改为 1024 后，受控调用成功返回可解析决策 JSON。是否对整体指标有改善仍需同条件对照。
- 同条件复跑四批：两批由千问决策，一批因输出结构无效降级规则；平均分与静态模型相同，消耗 4489 token。原 4000 token 预算出现单次跨额，已加入请求前余量检查并将此小规模配置预算设为 6000；这是模拟数据结果，不是赛题成绩。
- Gas 真实数据首轮在模型服务 HTTP 错误时自动回退到本地规则；详见《Gas真实数据初测》。这轮不能称作千问策略性能对照。
- 当天稍后 27B 恢复可用。改用官方采样 SDK 且关闭思考后，Gas 九批均由 27B 决策，平均 macro-F1 0.6241；完整证据见[Qwen27B 真实数据对照](Qwen27B真实数据对照.md)。

## 27B 曾无可用服务实例（历史）

2026-11-10 早些时候的诊断中，`Qwen/Qwen3.8-27B` 返回 HTTP 503 和 `no available server`；同一账号调用 `Qwen/Qwen3.5-4B` 返回 HTTP 200。后来 27B 已恢复。服务状态可变，重复充值或重装依赖不能解决这种服务端无实例状态。

双击 `check-pytrio.cmd` 可以进行一次短检查（会产生少量 token 用量）。成功后再做 Gas 千问对照；失败时继续使用本地规则。若持续返回 503，向平台反馈模型 ID、时间、HTTP 503 和 `no available server`，不要提供 API Key。官方联系入口见 https://docs.pytrio.com/docs 。
