模块接口总览
============

MASDiff 的可插拔能力通过一组抽象基类实现。用户只需继承对应基类并实现约定方法，即可通过 YAML 接入框架。

接口列表
--------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - 模块
     - 定义位置
     - 核心职责
   * - :doc:`environment`
     - ``src/environments/base.py``
     - 仿真、收集经验库和 Tau、产出评价数据
   * - :doc:`dqn`
     - ``src/dqn/base.py``
     - 初始化策略、构建训练数据、训练每个智能体
   * - :doc:`diffusion`
     - ``src/diffusion/base.py``
     - 随机初始化、生成奖励、训练扩散模型、截断扩散变异
   * - :doc:`evolution`
     - ``src/evolution/base.py``
     - 从种群中选择精英个体
   * - :doc:`metrics`
     - ``src/metrics/base.py``
     - 计算适应度指标 ρ
   * - :doc:`parallel`
     - ``src/parallel/base.py``
     - 并行（或串行）执行 map 任务

接口数据契约
------------

各模块之间通过以下数据对象传递信息：

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - 数据
     - 说明
   * - ``tau``
     - 仿真收集的环境状态条件，格式由用户约定，需在 Environment 和 DiffusionModel 之间保持一致
   * - ``rewards``
     - 扩散模型生成的奖励矩阵，格式需在 DiffusionModel 和 DqnModule 之间保持一致
   * - ``experience_buffers``
     - 每个智能体的经验库，格式需在 Environment 和 DqnModule 之间保持一致
   * - ``simulation_data``
     - 评估仿真的输出，格式需在 Environment 和 Metric 之间保持一致
   * - ``q``
     - 评价基准，格式需在 QProvider 和 Metric 之间保持一致

.. warning::

   框架只检查模块是否继承了正确的基类，不检查数据格式是否兼容。
   跨模块的数据契约（Tau 的形状、rewards 的维度等）需要用户自行保证一致。
