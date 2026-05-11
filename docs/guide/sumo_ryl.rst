内置示例：SUMO 路网仿真（sumo_ryl）
=====================================

MASDiff 提供了一套完整的 SUMO 路网仿真示例实现，用于交通路网车辆路径规划与排队优化场景。

场景说明
--------

- 环境：基于 SUMO + TraCI 的城市路网仿真（默认使用重庆路网）；
- 目标：通过优化 DQN 策略，使路网排队长度接近 A* 最优基线；
- 智能体：每辆车是一个智能体，使用 DQN 决策路径选择；
- Q 的含义：纯 A* 策略下的全网排队长度时序矩阵（评价基准）。

相关文件
--------

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - 文件
     - 说明
   * - ``src/environments/sumo_ryl.py``
     - SUMO 环境实现
   * - ``src/dqn/sumo_ryl_dqn.py``
     - DQN 模块实现
   * - ``src/diffusion/sumo_ryl_diffusion.py``
     - 扩散模型实现
   * - ``src/q/sumo_ryl_q.py``
     - Q 构建与缓存
   * - ``src/metrics/sumo_ryl_metric.py``
     - ρ 指标计算
   * - ``src/evolution/temperature_selection.py``
     - 温度采样精英选择
   * - ``configs/sumo_ryl.yaml``
     - 示例配置（M=100, N=1557）
   * - ``configs/sumo_ryl_4000.yaml``
     - 大规模配置（N=4000）

SumoRylEnvironment
------------------

仿真时，对每辆车首次出现时做路径规划：

- **有 policy**：使用"DQN 选分支 + A* 补全路径"；
- **无 policy（None）**：退化为纯 A*（按路段长度为代价）。

``simulate_collect`` 输出：

- ``experience_buffers``：长度 = ``num_car``，每条经验格式 ``[s, a, r]``，``r`` 留空；
- ``tau``：形状 ``[num_car, num_road, 2]``，两个特征为"到终点最短距离"和"当前路段排队长度"。

``simulate_evaluate`` 输出：

- ``simulation_data``：形状 ``[end_tick / sample_interval, num_road]``，表示各采样时刻全网排队长度。

SumoRylQProvider
----------------

- 若缓存文件存在（``outputs/q_sumo_ryl.pt``），直接加载；
- 否则以 ``policies=[None] * num_car`` 调用环境评估，得到纯 A* 基准下的 Q；
- 可选将结果缓存到磁盘，避免重复计算。

SumoRylMetric
-------------

计算 Q 与 simulation_data 的 MSE，并转换为越大越好的 ρ：

.. code-block:: text

   ρ = 1 / (1 + MSE)    # 默认 inv1p 模式

支持其他变换：``neg``、``inv``、``exp``。

SumoRylDqnModule
----------------

- 每个 agent 对应独立的 ``DQNModel``；
- 状态维度：2（到终点距离 + 当前排队长度）；
- 动作：下一条道路的索引；
- 训练方式：单步回归（用 ``(s, a, r)`` 拟合 ``Q(s, a)``）；
- 如果某 agent 经验数量少于 ``min_experiences``，直接返回未训练模型。

SumoRylDiffusionModel
---------------------

- 输入 ``tau`` 形状：``[num_car, num_road, 2]``；
- 输出 ``rewards`` 形状：``[num_car, num_road]``；
- 内部采用条件噪声预测网络（``TauDiffusionModel``）；
- 支持 DDPM / DDIM 采样、linear / cosine beta schedule。

TemperatureEliteSelector
------------------------

使用 softmax 温度采样选择精英，引入探索性避免早熟：

.. code-block:: text

   probs = softmax(temperature * rho_values)
   elites = multinomial_sample(probs, elite_count)

运行环境要求
------------

.. warning::

   运行 sumo_ryl 示例需满足以下条件：

   - 已安装 SUMO，且 ``SUMO_HOME/tools`` 可用于 Python 导入；
   - ``traci`` 和 ``sumolib`` 可正常使用；
   - 地图和路由文件路径与配置文件中一致；
   - 安装了 ``torch``；
   - ``sumo_ryl.yaml`` 默认 M=100, N=1557，需要充足的 CPU/GPU/内存和 Ray 集群支持。
