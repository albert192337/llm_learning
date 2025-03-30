### 关键要点
- 研究表明，P/D分离服务器是将大模型AI推理中的Prefill和Decode阶段分开处理，分别使用不同服务器优化资源利用率。
- 证据显示，Prefill阶段计算密集，适合高性能GPU；Decode阶段内存密集，可用较低成本硬件。
- 它似乎可能通过高效数据传输（如RDMA）降低延迟，微软的SplitWise实现显示吞吐量提高1.4倍，成本降低20%。

---

### 什么是P/D分离服务器？

**简介**  
P/D分离服务器是一种现代大模型AI推理优化架构，将推理过程分为两个阶段：Prefill（预填充或提示阶段）和Decode（解码或生成令牌阶段），并将它们分配到不同的服务器上处理。这种方法旨在根据每个阶段的计算需求优化硬件资源利用率。

**Prefill和Decode阶段**  
- **Prefill阶段**：处理输入提示，计算关键值（KV）缓存。这是计算密集型任务，需要高性能GPU来并行处理整个输入序列。  
- **Decode阶段**：顺序生成输出令牌，每次生成一个，依赖KV缓存访问，属于内存密集型任务，对计算能力要求较低。  

**为什么需要分离？**  
传统方法在同一硬件上处理两个阶段会导致资源浪费。例如，Decode阶段未充分利用GPU的计算能力，而内存带宽成为瓶颈。P/D分离允许使用高计算能力的GPU处理Prefill，使用大内存或成本较低的硬件处理Decode，从而降低成本并提高性能。

**一个意外的细节**  
微软的SplitWise技术不仅分离两个阶段，还包括一个混合池动态调整负载，KV缓存通过高速互联（如InfiniBand）传输，几乎不影响用户感知的延迟。

---

---

### 详细报告：现代大模型AI推理优化中的P/D分离服务器概念

#### 引言  
随着大语言模型（LLM）在生成式AI应用中的广泛使用，其推理过程对计算资源的需求日益增长。传统方法在单一硬件上处理推理的两个主要阶段——Prefill和Decode——往往导致资源利用率低、成本高、延迟大。为解决这些问题，P/D分离服务器（Prefill/Decode分离服务器）成为一种新兴优化架构。本报告将详细探讨其概念、实现、优势和挑战。

#### LLM推理的两个阶段  
LLM推理可分为以下两个阶段：  
- **Prefill阶段（提示阶段）**：模型处理用户输入的提示，计算注意力机制中的关键值（KV）缓存。这一阶段是计算密集型的，需要并行处理整个输入序列。例如，对于输入“告诉我一个故事”，模型会一次性处理所有提示令牌，生成第一个输出令牌。  
- **Decode阶段（生成令牌阶段）**：模型顺序生成后续输出令牌，每次生成一个新令牌，依赖之前计算的KV缓存。这一阶段是内存密集型的，每次操作需要访问和更新KV缓存，计算需求较低，但受内存带宽限制。  

研究表明，这两个阶段的计算特性截然不同：Prefill阶段充分利用GPU的并行计算能力，而Decode阶段更多依赖内存访问效率。

#### P/D分离的概念  
P/D分离服务器将Prefill和Decode阶段分配到不同的服务器或机器池，分别优化硬件资源。这种分离基于以下洞察：  
- Prefill阶段需要高计算能力的GPU，如最新一代NVIDIA A100，以快速处理大批量输入。  
- Decode阶段对计算能力要求较低，可以使用较旧的GPU或大内存硬件，降低成本。  

通过这种方式，P/D分离服务器实现阶段特定的资源管理，改善整体系统效率。例如，微软的SplitWise技术将推理请求分为两个阶段，分别在不同机器池上运行，并引入一个混合池动态处理负载。

#### 实现细节  
P/D分离的实现涉及以下关键方面：  
- **硬件选择**：Prefill服务器配备高性能GPU，优化计算密集型任务；Decode服务器优先考虑大内存或高带宽硬件，满足内存密集型需求。  
- **数据传输**：Prefill阶段计算的KV缓存需高效传输到Decode服务器。研究显示，使用远程直接内存访问（RDMA）技术和高速互联（如InfiniBand）可确保低延迟传输。例如，微软SplitWise通过InfiniBand传输KV缓存，几乎不影响用户感知的延迟。  
- **调度和批处理**：需要先进的调度策略管理不同阶段的负载。例如，SplitWise包括一个混合池，根据实时需求动态调整资源分配。  

#### 案例分析：微软SplitWise  
微软的SplitWise是P/D分离的一个典型实现，论文“Splitwise: Efficient generative LLM inference using phase splitting”[Splitwise: Efficient generative LLM inference using phase splitting](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/)详细描述了其设计。  
- **架构**：SplitWise将推理分为提示计算和生成令牌两个阶段，分别使用专用机器池，并设置一个混合池处理动态需求。  
- **性能**：研究显示，SplitWise在相同成本和功耗预算下，吞吐量提高2.35倍；在20%较低成本下，吞吐量提高1.4倍。  
- **应用**：适用于Azure等云服务环境，优化GPU集群利用率，降低运营成本。  

另一个案例是Mooncake（Kimi开源的推理架构），也采用了类似的分离策略，展示了异构集群布局和负载均衡的效率。

#### 优势和性能提升  
P/D分离服务器带来以下优势：  
- **资源利用率提高**：通过匹配硬件特性与阶段需求，减少资源浪费。例如，Decode阶段可使用成本较低的硬件，降低整体开支。  
- **成本效率**：研究表明，分离后可显著降低硬件成本。例如，Sohu文章提到在RTP-LLM场景中，实例数量减少24%，平均延迟降低48%，P99延迟降低78%[分布式推理技术新纪元：大模型P-D分离启动](https://www.sohu.com/a/851451367_122004016)。  
- **性能优化**：通过高效数据传输和阶段优化，降低端到端延迟，提高吞吐量。  

#### 挑战和争议  
尽管P/D分离有显著优势，但也面临挑战：  
- **数据传输开销**：KV缓存的传输需确保低延迟，高带宽互联（如InfiniBand）成本较高。  
- **复杂性增加**：需要先进的调度和负载均衡策略，增加系统设计难度。  
- **硬件依赖**：对高速互联和异构硬件的支持要求较高，可能限制某些场景的应用。  

争议在于，是否所有场景都适合P/D分离。例如，对于短提示和少量生成令牌的任务，分离可能引入不必要的开销。

#### 表格：P/D分离与传统方法的比较  

| 方面               | 传统方法                     | P/D分离服务器                     |
|--------------------|------------------------------|-----------------------------------|
| 硬件利用           | 单一硬件处理两阶段，资源浪费 | 阶段专用硬件，资源匹配度高         |
| 成本               | 高，需高性能GPU全程支持     | 低，Decode可使用低成本硬件       |
| 延迟               | 较高，Decode阶段受限         | 低，RDMA传输优化，延迟显著降低   |
| 吞吐量             | 受限于单一硬件能力           | 提高，例如SplitWise吞吐量提高2.35倍 |
| 复杂性             | 较低，单一架构               | 较高，需调度和数据传输优化         |

#### 未来趋势  
随着LLM规模扩大和推理需求增长，P/D分离服务器有望进一步发展。研究者正探索更高效的传输协议、异构硬件整合和自动化调度策略，以降低成本和复杂性。

#### 结论  
P/D分离服务器是现代大模型AI推理优化的重要创新，通过将Prefill和Decode阶段分离，优化资源利用率，降低成本并提升性能。微软SplitWise等案例展示了其潜力，但也需解决数据传输和复杂性挑战。

#### 关键引用  
- [Splitwise: Efficient generative LLM inference using phase splitting](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/)  
- [Splitwise improves GPU usage by splitting LLM inference phases - Microsoft Research](https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/)  
- [LLM Inference Optimization — Splitwise | by Don Moon | Sep, 2024 | Medium](https://medium.com/byte-sized-ai/llm-inference-optimization-3-splitwise-88dfcf0948ca)  
- [分布式推理技术新纪元：大模型P-D分离启动](https://www.sohu.com/a/851451367_122004016)