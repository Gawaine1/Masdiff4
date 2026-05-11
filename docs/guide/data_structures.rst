核心数据结构
============

Individual
----------

定义位置：``src/pipeline/types.py``

``Individual`` 是种群的基本单元，代表算法中的一个候选解：

.. code-block:: python

   @dataclass
   class Individual:
       tau: Any                        # 环境收集到的条件向量
       rewards: Any                    # 扩散模型生成的奖励
       rho: float                      # 适应度指标，越大越好
       experience_buffers: list[list[Any]]  # 每个智能体的经验库
       policies: list[Any]             # 策略列表（DQN 模型）
       metadata: dict[str, Any]        # 附加信息

各字段说明：

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - 字段
     - 说明
   * - ``tau``
     - 环境仿真时收集的状态条件信息，作为扩散模型生成奖励的条件。具体结构由用户实现决定（例如 SUMO 场景中为形状 ``[num_car, num_road, 2]`` 的张量）
   * - ``rewards``
     - 扩散模型生成的奖励矩阵 ``R``，用于填充经验库中的 ``r`` 字段
   * - ``rho``
     - 适应度/评价指标，越大越好。由 ``Metric.compute_rho()`` 计算得出
   * - ``experience_buffers``
     - 长度等于智能体数量 ``N``，每个元素是该智能体的经验序列（``[s, a, r]`` 列表）
   * - ``policies``
     - 训练后的 DQN 策略列表，一个智能体对应一个策略。为减少内存压力，当前实现中通常存空列表
   * - ``metadata``
     - 附加信息字典，框架会在其中记录 ``initial_index``、``iteration_k``、``parent_rho``、``simulation_data`` 等调试信息

.. note::

   为避免内存和序列化膨胀，``runner.py`` 当前实现中 ``policies`` 通常存 ``[]``。
   ``Individual`` 结构支持保存策略，但主流程主动避免保存大模型对象。

Population
----------

.. code-block:: python

   Population = list[Individual]

种群即个体列表，是进化迭代的操作对象。

MasDiffConfig
-------------

定义位置：``src/config/schema.py``

配置对象，由 YAML 文件加载后得到，关键字段见下表：

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - 字段
     - 说明
   * - ``seed``
     - 随机种子
   * - ``algorithm.M``
     - 种群规模
   * - ``algorithm.N``
     - 每个个体中的智能体数量
   * - ``algorithm.K``
     - 进化迭代次数
   * - ``elite.elite_count``
     - 每轮选择的精英数量
   * - ``truncated_diffusion.add_noise_steps``
     - 截断扩散加噪步数
   * - ``truncated_diffusion.denoise_steps``
     - 截断扩散去噪步数
   * - ``logging.best_rho_csv_path``
     - 最优 ρ 输出路径
   * - ``q_provider / environment / dqn_module / ...``
     - 各模块的 ``ModuleSpec``（class_path + kwargs）

经验条目格式
------------

单条经验的格式约定为：

.. code-block:: text

   [s, a, r]

其中：

- ``s``：状态，格式为 ``[当前位置标识, [特征1, 特征2, ...]]``，特征维度通过配置设定；
- ``a``：动作（如下一条道路的索引）；
- ``r``：奖励，仿真收集阶段留空，由扩散模型生成的 ``R`` 后续填充。

Tau 格式约定
------------

``Tau`` 是从仿真中采集的环境状态特征，作为扩散模型的生成条件。

格式约定（以 SUMO 场景为例）：

.. code-block:: text

   形状：[num_car, num_road, 2]
   两个特征维度：
     - 当前路到终点的最短距离
     - 当前路段排队长度
