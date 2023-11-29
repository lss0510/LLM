| **部署框架** | **适配器微调** | **量化** | **显存优化** | **批处理** | **多GPU并行** | **算子优化** | **硬件支持** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM v0.2.2 | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ❌ | NVIDIA GPU |
| Hugging Face TGI v1.1.1 | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ❌ | NVIDIA GPU |
| Faster Transformer | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | NVIDIA GPU |
| TensorRT-LLM | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | NVIDIA GPU |
| CTranslate2 | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | NVIDIA GPU/x86-64 CPU/ARM64 CPU |
| DeepSpeed-MII | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | NVIDIA GPU |
| OpenLLM | ✔️ | ✔️ | ❌ | ✔️ | ❌ | ❌ | GPU |
| Ray Serve | ❌ | ✔️ | ❌ | ✔️ | ✔️ | ❌ | GPU |
| MLC LLM | ❌ | ✔️ | ✔️ | ❌ | ✔️ | ✔️ | Inter/AMD/NVIDIA/APPLE/ APPLE A-series /Adreno/Mali GPU |

Ps：以上表格总结内容均来自官方github文档。
# vLLM
该部署框架由uc伯克利开源，其核心算法是**PagedAttention**（一种新的注意力算法，可有效管理注意力键和值缓存)大大提升了Memory的利用率，使得吞吐量变大，多batch的推理速度变快）、**Continuous Batching**（连续批处理，一种更合理的系统级别批处理优化方法，提升GPU使用效率，保证多个batch之间相互独立，不用彼此等待结束，提升多batch推理速度）。
# Hugging Face TGI
目前已经支持PagedAttention、Continuous Batching。
目前在做AMD GPU支持。
# Faster Transformer
Faster Transformer（FT）是一个库，用于实现基于Transformer的神经网络推理的加速引擎，对于大模型，其以分布式方式跨越许多 GPU 和节点。Faster Transformer包含Transformer块的高度优化版本的实现，其中包含编码器Encoder和解码器Decoder部分。基于 FT 可以运行完整的编码器-解码器架构（如 T5 大模型）以及仅编码器模型（如 BERT）或仅解码器模型（如 GPT）的推理。
[参考资料](https://zhuanlan.zhihu.com/p/626303702)
# TensorRT-LLM
NVIDLA开源的大模型推理提速框架，具有以下特点：

- 飞行批处理
- FP8数据格式支持
- 广泛的模型支持
- 并行化和分布式推理
- 内核优化

[参考资料](https://zhuanlan.zhihu.com/p/655414711)
# CTranslate2
CTranslate2是一个C++和Python库，用于使用Transformer模型进行高效推理，相比于原生的pytorch框架推理，速度提升很多。
[参考资料](https://zhuanlan.zhihu.com/p/609374407)
# DeepSpeed-MII 
该部署框架由微软开发，是DeepSpeed的一个开源 Python 库。具有以下几点优势：

- 显著降低了推理时延迟和成本；
- 支持基于DeepSpeed-Inference的一系列广泛优化：transformers 的深度融合、用于多 GPU 推理的自动张量切片、使用 ZeroQuant 进行动态量化、适用于资源受限情况下的ZeRO推理、编译器优化等；

[参考资料](https://zhuanlan.zhihu.com/p/660817458)
# OpenLLM
OpenLLM是一个用于在生产中操作大型语言模型 (LLM) 的开放平台，可以轻松地微调、服务、部署和监控任何LLM大模型。具有以下几点优势：

- 灵活的API支持
- 方便用户自己构建合适的大模型应用程序
- 简化部署（服务器镜像等）
- 自带微调工具

[参考资料](https://zhuanlan.zhihu.com/p/651130017)
# Ray Serve
Ray Serve是一个可扩展的模型服务库，用于构建在线推理API。Serve与框架无关，因此可以使用一个工具包来为深度学习模型的所有内容提供服务。我个人的感觉，这种像是服务器平台的部署框架，好像不适用于边端部署。
[参考资料](https://zhuanlan.zhihu.com/p/653352979)
# MLC LLM
LLM的机器学习编译（MLC LLM）是一种通用的部署解决方案，它使LLM能够利用本机硬件加速在消费者设备上高效运行。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701247850491-64e9c294-e33d-4b2e-9598-f7aee585ebce.png#averageHue=%23d8d6a9&clientId=ua7805fae-ae5f-4&from=paste&id=ua65c87f1&originHeight=368&originWidth=1080&originalType=url&ratio=1.25&rotation=0&showTitle=false&size=328676&status=done&style=none&taskId=ucb6ee8bf-982c-469c-a258-bcacedffb0d&title=)
