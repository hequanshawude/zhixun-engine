# 2026-11-20 r4 参赛镜像源码

`app/` 包含 `zhixun-engine:b-anon-r4-20261120` 最终运行的 11 个源码文件。镜像使用赛事 SDK 1.0.2 处理 A、B、C 三组续训与逐行预测，大模型决策经宿主网关发给 Qwen/Qwen3.8-27B。

- A、C 从实际选定的父检查点及训练状态续训，读取宿主回包中的验证结果，并在候选退步时保留父模型。
- B 使用训练集的振动特征和剩余寿命标签。样本可分组时按轴承拟合教师。样本匿名时按训练标签加权拟合带岭正则的残差教师，再用 Adam 更新学生模型的读出层。
- 每轮采用宿主给定的输出上限，关闭思考模式。正文为空或 JSON 无效时补问一次。实验理由按 UTF-8 字节数裁剪，留出验收说明所需空间。

B 组在相同的匿名离线回放中，RMSE 从 666.88 降至 599.52 分钟。三根公开验证轴承的误差均下降，10 轮后保留第 9 轮候选。该对照使用公开数据和模拟决策。实际模型实验记录可在 [SwanLab 项目](https://swanlab.cn/@hequan/zhixunyinqing) 查询。赛事成绩以平台评测为准。

## 构建和核验

先按 [11 月 18 日版本的构建说明](../20261118/README.md#构建与校验) 准备 `zhixun-engine:sdk-1.0.2-local-20261118`，再在本目录构建 Linux/amd64 镜像。

```sh
docker build --platform linux/amd64 -t zhixun-engine:b-anon-r4-20261120 .
docker run --rm --network none --read-only zhixun-engine:b-anon-r4-20261120 describe
docker save -o zhixun-engine-b-anon-r4-20261120.tar zhixun-engine:b-anon-r4-20261120
```

`IMAGE-SOURCE.sha256` 记录已核对的 11 个源码文件。原始交付归档的 SHA-256 为 `db0c58f54636e9c4e51e654ba1bc7ded5ff3bd11e3012443367a8800dfa19c1b`，大小为 64,640,512 字节。重新构建的归档层和哈希可能不同，可用源码校验清单和 `describe` 检查内容。
